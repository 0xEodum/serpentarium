import numpy as np
from typing import Dict, Any, Optional, Union, Tuple


class BaseModel:
    """Базовый класс для всех моделей"""

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Выбор действия на основе состояния с возможностью exploration

        Args:
            state: Текущее состояние
            epsilon: Вероятность случайного действия (exploration)

        Returns:
            int: Выбранное действие
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Обновление модели на основе опыта

        Returns:
            Dict[str, Any]: Метрики обучения
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def save(self, path: str) -> None:
        """
        Сохранение модели

        Args:
            path: Путь для сохранения
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def load(self, path: str) -> bool:
        """
        Загрузка модели

        Args:
            path: Путь к файлу модели

        Returns:
            bool: Успех загрузки
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")