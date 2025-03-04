from typing import Tuple, Dict, Any, List, Union
import numpy as np


class BaseEnvironment:
    """Базовый интерфейс для игровых сред"""

    def reset(self) -> np.ndarray:
        """
        Сброс среды и возврат начального состояния

        Returns:
            np.ndarray: Начальное состояние
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Выполнение действия

        Args:
            action: Выбранное действие

        Returns:
            Tuple[np.ndarray, float, bool]:
                - Новое состояние
                - Награда
                - Флаг завершения эпизода
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def get_state(self) -> np.ndarray:
        """
        Получение текущего состояния

        Returns:
            np.ndarray: Текущее состояние
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def get_state_shape(self) -> Tuple[int, ...]:
        """
        Возвращает форму состояния для инициализации сети

        Returns:
            Tuple[int, ...]: Форма состояния
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def get_action_space(self) -> Dict[str, Any]:
        """
        Возвращает информацию о пространстве действий

        Returns:
            Dict[str, Any]: Информация о пространстве действий
                - 'n': Количество действий
                - 'actions': Список или словарь с описанием действий
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def render(self) -> None:
        """
        Отображение текущего состояния среды (для отладки)
        """
        pass