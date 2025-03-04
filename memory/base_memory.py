from typing import Tuple, List, Any, Optional, Dict
import numpy as np


class BaseMemory:
    """Базовый интерфейс для всех видов памяти"""

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
        raise NotImplementedError("Метод должен быть реализован в подклассе")

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
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def __len__(self) -> int:
        """
        Возвращает количество переходов в памяти

        Returns:
            int: Количество переходов
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def save(self, path: str) -> None:
        """
        Сохранение памяти в файл

        Args:
            path: Путь для сохранения
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def load(self, path: str) -> bool:
        """
        Загрузка памяти из файла

        Args:
            path: Путь к файлу

        Returns:
            bool: Успех загрузки
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")