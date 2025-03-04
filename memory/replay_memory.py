import pickle
import os
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Any, Optional, Dict, Deque
from memory.base_memory import BaseMemory


class ReplayMemory(BaseMemory):
    """Реализация простой памяти воспроизведения для обучения с подкреплением"""

    def __init__(self, capacity: int):
        """
        Инициализация памяти воспроизведения

        Args:
            capacity: Максимальное количество переходов в памяти
        """
        self.memory: Deque = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Добавление перехода в память

        Args:
            state: Текущее состояние
            action: Выбранное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[List[np.ndarray], List[int],
    List[float], List[np.ndarray], List[bool]]:
        """
        Выборка пакета переходов из памяти

        Args:
            batch_size: Размер пакета

        Returns:
            Tuple[List[np.ndarray], List[int], List[float], List[np.ndarray], List[bool]]:
                - Список состояний
                - Список действий
                - Список наград
                - Список следующих состояний
                - Список флагов завершения
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Возвращает количество переходов в памяти

        Returns:
            int: Количество переходов
        """
        return len(self.memory)

    def save(self, path: str) -> None:
        """
        Сохранение памяти в файл

        Args:
            path: Путь для сохранения
        """
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, path: str) -> bool:
        """
        Загрузка памяти из файла

        Args:
            path: Путь к файлу

        Returns:
            bool: Успех загрузки
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)
            return True
        except Exception as e:
            print(f"Ошибка при загрузке памяти: {e}")
            return False