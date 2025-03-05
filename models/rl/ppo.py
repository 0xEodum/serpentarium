import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from typing import Dict, List, Any, Tuple, Optional, Union

from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from config.network_config import NetworkConfig
from config.training_config import TrainingConfig


class PPO(BaseModel):
    """Реализация алгоритма Proximal Policy Optimization (PPO)"""

    def __init__(self,
                 env: BaseEnvironment,
                 network_config: NetworkConfig,
                 training_config: TrainingConfig,
                 memory: Optional[Any] = None,  # Не используется в PPO, для совместимости интерфейса
                 device: Optional[torch.device] = None):
        """
        Инициализация модели PPO

        Args:
            env: Окружение
            network_config: Конфигурация нейронной сети
            training_config: Конфигурация обучения
            memory: Не используется в PPO (для совместимости с интерфейсом)
            device: Устройство для вычислений (CPU/GPU)
        """
        self.env = env
        self.network_config = network_config
        self.training_config = training_config

        # Устанавливаем устройство
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Получаем размеры входных данных и количество действий
        self.input_shape = env.get_state_shape()
        self.action_space = env.get_action_space()
        self.n_actions = self.action_space['n']

        # Настраиваем конфигурацию сети, если необходимо
        if network_config.input_shape is None:
            network_config.input_shape = self.input_shape
        if network_config.output_size is None:
            network_config.output_size = self.n_actions

        # Создаем сети актора и критика
        from models.network_builder import PyTorchNetworkBuilder

        # Создаем копию конфигурации для сети критика с одним выходом
        value_network_config = NetworkConfig()
        value_network_config.__dict__.update(network_config.__dict__)
        value_network_config.output_size = 1

        self.actor = PyTorchNetworkBuilder.build_network(network_config).to(self.device)
        self.critic = PyTorchNetworkBuilder.build_network(value_network_config).to(self.device)

        # Создаем оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=training_config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=training_config.critic_learning_rate)

        # Параметры PPO
        self.clip_param = training_config.ppo_clip_param
        self.ppo_epochs = training_config.ppo_epochs
        self.num_mini_batches = training_config.ppo_mini_batches
        self.value_coef = training_config.value_loss_coef
        self.entropy_coef = training_config.entropy_coef

        # Буфер для сбора опыта
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # Счетчик шагов
        self.steps_done = 0
        self.max_steps = training_config.ppo_steps_per_update

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Выбор действия на основе состояния

        Args:
            state: Текущее состояние
            epsilon: Не используется в PPO (для совместимости с интерфейсом)

        Returns:
            int: Выбранное действие
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(state_tensor)
            value = self.critic(state_tensor)

            # Создаем распределение вероятностей
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)

            # Выбираем действие
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Если мы в режиме сбора опыта, сохраняем переход
            if len(self.states) < self.max_steps:
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())

            return action.item()

    def update(self) -> Dict[str, Any]:
        """
        Обновление модели на основе собранного опыта

        Returns:
            Dict[str, Any]: Метрики обучения
        """
        # Если не собрано достаточно опыта, пропускаем обновление
        if len(self.states) < self.max_steps:
            return {"policy_loss": None, "value_loss": None, "entropy": None}

        # Преобразуем данные в тензоры
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)

        # Вычисляем преимущества и целевые значения
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Обновляем сети актора и критика через несколько эпох
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.ppo_epochs):
            # Генерируем пермутацию индексов
            indices = torch.randperm(states.size(0))

            # Разбиваем на мини-батчи
            batch_size = states.size(0) // self.num_mini_batches
            if batch_size == 0:
                batch_size = 1

            for start_idx in range(0, states.size(0), batch_size):
                # Получаем индексы для текущего мини-батча
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = indices[start_idx:end_idx]

                # Получаем батч
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Получаем новые логиты и значения
                logits = self.actor(batch_states)
                values = self.critic(batch_states).squeeze(-1)

                # Вычисляем новые вероятности
                probs = F.softmax(logits, dim=1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Вычисляем отношение новой и старой вероятностей
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Вычисляем суррогатные функции
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages

                # Вычисляем потери
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)

                # Общая потеря с учетом энтропии
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Обновляем сети
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # Ограничиваем градиенты для стабильности обучения
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        # Очищаем буфер
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # Возвращаем метрики
        return {
            "policy_loss": np.mean(policy_losses) if policy_losses else None,
            "value_loss": np.mean(value_losses) if value_losses else None,
            "entropy": np.mean(entropy_losses) if entropy_losses else None
        }

    def add_experience(self, reward: float, done: bool) -> None:
        """
        Добавление награды и флага завершения в буфер

        Args:
            reward: Полученная награда
            done: Флаг завершения эпизода
        """
        self.rewards.append(reward)
        self.dones.append(float(done))

        # Увеличиваем счетчик шагов
        self.steps_done += 1

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     dones: torch.Tensor, gamma: float = 0.99,
                     lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление Generalized Advantage Estimation (GAE)

        Args:
            rewards: Тензор наград
            values: Тензор значений из сети критика
            dones: Тензор флагов завершения
            gamma: Коэффициент дисконтирования
            lam: Параметр lambda для GAE

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Возвраты и преимущества
        """
        gae = 0
        returns = []
        advantages = []

        # Получаем последнее значение из сети критика
        with torch.no_grad():
            if len(self.states) > 0:
                state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
                next_value = self.critic(state_tensor).squeeze().item()
            else:
                next_value = 0

        # Расчет GAE в обратном порядке
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]

            delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
            gae = delta + gamma * lam * next_non_terminal * gae

            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)

        # Нормализация преимуществ
        advantages = torch.FloatTensor(advantages).to(self.device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return torch.FloatTensor(returns).to(self.device), advantages

    def save(self, path: str) -> None:
        """
        Сохранение модели

        Args:
            path: Путь для сохранения
        """
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Сохраняем модель
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load(self, path: str) -> bool:
        """
        Загрузка модели

        Args:
            path: Путь к файлу модели

        Returns:
            bool: Успех загрузки
        """
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.steps_done = checkpoint['steps_done']

            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели PPO: {e}")
            return False