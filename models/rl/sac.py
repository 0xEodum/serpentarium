import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import random
from typing import Dict, List, Any, Tuple, Optional, Union

from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from memory.base_memory import BaseMemory
from memory.replay_memory import ReplayMemory
from config.network_config import NetworkConfig
from config.training_config import TrainingConfig


class SAC(BaseModel):
    """Реализация алгоритма Soft Actor-Critic (SAC) для дискретных действий"""

    def __init__(self,
                 env: BaseEnvironment,
                 network_config: NetworkConfig,
                 training_config: TrainingConfig,
                 memory: Optional[BaseMemory] = None,
                 device: Optional[torch.device] = None):
        """
        Инициализация модели SAC

        Args:
            env: Окружение
            network_config: Конфигурация нейронной сети
            training_config: Конфигурация обучения
            memory: Объект памяти (создается автоматически, если не указан)
            device: Устройство для вычислений (CPU/GPU)
        """
        self.env = env
        self.network_config = network_config
        self.training_config = training_config

        # Устанавливаем устройство
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создаем или используем предоставленную память
        self.memory = memory if memory is not None else ReplayMemory(training_config.memory_size)

        # Получаем размеры входных данных и количество действий
        self.input_shape = env.get_state_shape()
        self.action_space = env.get_action_space()
        self.n_actions = self.action_space['n']

        # Настраиваем конфигурацию сети, если необходимо
        if network_config.input_shape is None:
            network_config.input_shape = self.input_shape
        if network_config.output_size is None:
            network_config.output_size = self.n_actions

        # Создаем копию конфигурации для Q-сетей и сети ценности
        q_network_config = NetworkConfig()
        q_network_config.__dict__.update(network_config.__dict__)
        q_network_config.output_size = self.n_actions

        value_network_config = NetworkConfig()
        value_network_config.__dict__.update(network_config.__dict__)
        value_network_config.output_size = 1

        # Создаем сети актора и критиков
        from models.network_builder import PyTorchNetworkBuilder

        # Сеть актора (policy network)
        self.actor = PyTorchNetworkBuilder.build_network(network_config).to(self.device)

        # Две Q-сети
        self.qf1 = PyTorchNetworkBuilder.build_network(q_network_config).to(self.device)
        self.qf2 = PyTorchNetworkBuilder.build_network(q_network_config).to(self.device)

        # Целевые Q-сети
        self.target_qf1 = PyTorchNetworkBuilder.build_network(q_network_config).to(self.device)
        self.target_qf2 = PyTorchNetworkBuilder.build_network(q_network_config).to(self.device)

        # Инициализируем целевые сети
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        # Устанавливаем целевые сети в режим оценки
        self.target_qf1.eval()
        self.target_qf2.eval()

        # Создаем оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=training_config.learning_rate)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=training_config.learning_rate)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=training_config.learning_rate)

        # Параметры SAC
        self.gamma = training_config.gamma

        # Коэффициент температуры (автоматически настраиваемый или фиксированный)
        self.target_entropy = -np.log(1.0 / self.n_actions) * 0.98  # Целевая энтропия
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=training_config.learning_rate)

        # Параметр обновления целевой сети (использование мягкого обновления)
        self.tau = 0.005  # Параметр плавного обновления целевых сетей

        # Счетчики
        self.steps_done = 0
        self.update_counter = 0

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Выбор действия на основе состояния с возможностью exploration

        Args:
            state: Текущее состояние
            epsilon: Вероятность случайного действия (exploration)

        Returns:
            int: Выбранное действие
        """
        # Используем epsilon exploration для совместимости с интерфейсом BaseModel
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(state_tensor)

            # Получаем вероятности действий
            action_probs = F.softmax(logits, dim=1)

            # В режиме оценки (inference) выбираем действие с наибольшей вероятностью
            if epsilon == 0:  # если находимся в режиме оценки
                action = torch.argmax(action_probs, dim=1).item()
            else:
                # В режиме обучения используем стохастическую политику
                dist = Categorical(action_probs)
                action = dist.sample().item()

            return action

    def update(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Обновление модели на основе опыта из памяти

        Args:
            batch_size: Размер пакета (если None, используется из конфигурации)

        Returns:
            Dict[str, Any]: Метрики обучения
        """
        # Если недостаточно опыта в памяти, пропускаем обновление
        batch_size = batch_size if batch_size is not None else self.training_config.batch_size
        if len(self.memory) < batch_size:
            return {"actor_loss": None, "qf1_loss": None, "qf2_loss": None, "alpha_loss": None, "alpha": None}

        # Выбираем пакет переходов из памяти
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Преобразуем данные в тензоры
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Текущее значение alpha
        alpha = self.log_alpha.exp().item()

        # 1. Обновляем Q-функции
        q1_values = self.qf1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_values = self.qf2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Вычисляем целевые Q-значения
        with torch.no_grad():
            # Получаем логиты следующего состояния и вероятности действий
            next_logits = self.actor(next_states)
            next_action_probs = F.softmax(next_logits, dim=1)
            next_log_probs = F.log_softmax(next_logits, dim=1)

            # Энтропия следующего состояния
            entropy = -torch.sum(next_action_probs * next_log_probs, dim=1)

            # Q-значения для всех действий в следующем состоянии
            next_q1 = self.target_qf1(next_states)
            next_q2 = self.target_qf2(next_states)

            # Берем минимум из двух Q-значений
            next_q_min = torch.min(next_q1, next_q2)

            # Ожидаемое Q-значение с учетом вероятности действий
            expected_next_q = (next_action_probs * (next_q_min - alpha * next_log_probs)).sum(dim=1)

            # Целевое Q-значение с учетом награды и дисконтирования
            target_q = rewards + (1 - dones) * self.gamma * expected_next_q

        # Вычисляем функции потерь для Q-сетей
        qf1_loss = F.mse_loss(q1_values, target_q)
        qf2_loss = F.mse_loss(q2_values, target_q)

        # Обновляем Q-сети
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # 2. Обновляем policy сеть (актор)
        logits = self.actor(states)
        action_probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Получаем Q-значения для текущих состояний
        with torch.no_grad():
            q1 = self.qf1(states)
            q2 = self.qf2(states)
            q_min = torch.min(q1, q2)

        # Вычисляем энтропию
        entropy = -torch.sum(action_probs * log_probs, dim=1)

        # Вычисляем функцию потерь для актора (с регуляризацией энтропии)
        inside_term = alpha * log_probs - q_min
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()

        # Обновляем актора
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Обновляем коэффициент температуры (alpha)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 4. Мягкое обновление целевых Q-сетей
        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        # Увеличиваем счетчик обновлений
        self.update_counter += 1

        return {
            "actor_loss": actor_loss.item(),
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha
        }

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
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'target_qf1': self.target_qf1.state_dict(),
            'target_qf2': self.target_qf2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'qf1_optimizer': self.qf1_optimizer.state_dict(),
            'qf2_optimizer': self.qf2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'update_counter': self.update_counter
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
            self.qf1.load_state_dict(checkpoint['qf1'])
            self.qf2.load_state_dict(checkpoint['qf2'])
            self.target_qf1.load_state_dict(checkpoint['target_qf1'])
            self.target_qf2.load_state_dict(checkpoint['target_qf2'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.qf1_optimizer.load_state_dict(checkpoint['qf1_optimizer'])
            self.qf2_optimizer.load_state_dict(checkpoint['qf2_optimizer'])

            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

            self.steps_done = checkpoint['steps_done']
            self.update_counter = checkpoint['update_counter']

            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели SAC: {e}")
            return False