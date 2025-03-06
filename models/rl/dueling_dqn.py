import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Any, Tuple, Optional, Union

from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from memory.base_memory import BaseMemory
from memory.replay_memory import ReplayMemory
from config.network_config import NetworkConfig
from config.training_config import TrainingConfig


class DuelingDQN(BaseModel):
    """Реализация алгоритма Dueling DQN"""

    def __init__(self,
                 env: BaseEnvironment,
                 network_config: NetworkConfig,
                 training_config: TrainingConfig,
                 memory: Optional[BaseMemory] = None,
                 device: Optional[torch.device] = None):
        """
        Инициализация модели Dueling DQN

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

        # Активируем использование Dueling архитектуры в конфигурации
        network_config.use_dueling_architecture = True

        # Создаем основную и целевую сети с Dueling архитектурой
        from models.network_builder import PyTorchNetworkBuilder
        self.policy_net = PyTorchNetworkBuilder.build_network(network_config).to(self.device)
        self.target_net = PyTorchNetworkBuilder.build_network(network_config).to(self.device)

        # Синхронизируем целевую сеть с основной
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Целевая сеть только для предсказания, не для обучения

        # Создаем оптимизатор
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=training_config.learning_rate)

        # Счетчик шагов обучения
        self.steps_done = 0

        # Счетчик для обновления целевой сети
        self.target_update_counter = 0

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Выбор действия на основе состояния с возможностью exploration

        Args:
            state: Текущее состояние
            epsilon: Вероятность случайного действия (exploration)

        Returns:
            int: Выбранное действие
        """
        # Проверяем, нужно ли выбирать случайное действие
        if random.random() < epsilon:
            # Случайное действие
            return random.randrange(self.n_actions)

        # Преобразуем состояние в тензор и получаем предсказание от сети
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            # Выбираем действие с максимальным Q-значением
            return q_values.max(1)[1].item()

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
            return {"loss": None}

        # Выбираем пакет переходов из памяти
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Преобразуем данные в тензоры
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Получаем текущие Q-значения для выбранных действий
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: выбираем действия с policy_net, но оцениваем их с target_net
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # Учитываем терминальные состояния
        next_q_values = next_q_values * (1 - dones)

        # Вычисляем целевые Q-значения
        target_q_values = rewards + (self.training_config.gamma * next_q_values)

        # Вычисляем функцию потерь (Smooth L1 Loss / Huber Loss)
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Оптимизируем модель
        self.optimizer.zero_grad()
        loss.backward()

        # Ограничиваем градиенты для стабильности обучения
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # Увеличиваем счетчик для обновления целевой сети
        self.target_update_counter += 1

        # Обновляем целевую сеть, если пришло время
        if self.target_update_counter % self.training_config.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_update_counter = 0

        return {"loss": loss.item()}

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
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'target_update_counter': self.target_update_counter
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

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            self.target_update_counter = checkpoint['target_update_counter']

            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return False