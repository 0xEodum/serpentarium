from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import time
from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from config.training_config import TrainingConfig


class BaseTrainer:
    """Базовый класс для всех тренеров"""

    def __init__(self,
                 model: BaseModel,
                 env: BaseEnvironment,
                 config: TrainingConfig):
        """
        Инициализация тренера

        Args:
            model: Модель для обучения
            env: Окружение
            config: Конфигурация обучения
        """
        self.model = model
        self.env = env
        self.config = config

        # Метрики обучения
        self.scores = []
        self.avg_scores = []
        self.metrics = {}

        # Флаг для отслеживания состояния обучения
        self.is_training = False

        # Время начала обучения
        self.start_time = None

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Запуск обучения

        Args:
            num_episodes: Количество эпизодов (если None, используется из конфигурации)

        Returns:
            Dict[str, Any]: Результаты обучения
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Оценка модели

        Args:
            num_episodes: Количество эпизодов для оценки

        Returns:
            Dict[str, Any]: Результаты оценки
        """
        raise NotImplementedError("Метод должен быть реализован в подклассе")

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Сохранение модели

        Args:
            path: Путь для сохранения (если None, используется из конфигурации)
        """
        path = path if path is not None else self.config.model_save_path
        self.model.save(path)
        print(f"Модель сохранена: {path}")

    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Загрузка модели

        Args:
            path: Путь к файлу модели (если None, используется из конфигурации)

        Returns:
            bool: Успех загрузки
        """
        path = path if path is not None else self.config.model_save_path
        result = self.model.load(path)
        if result:
            print(f"Модель загружена: {path}")
        else:
            print(f"Не удалось загрузить модель: {path}")
        return result

    def visualize_results(self) -> None:
        """
        Визуализация результатов обучения
        """
        from serpentarium.utils.visualization import Visualizer

        # Отображаем графики счета
        if self.scores:
            Visualizer.plot_scores(self.scores, self.avg_scores)

        # Отображаем метрики обучения
        if self.metrics:
            Visualizer.plot_metrics(self.metrics)

    def _update_metrics(self, episode_metrics: Dict[str, Any]) -> None:
        """
        Обновление метрик обучения

        Args:
            episode_metrics: Метрики за эпизод
        """
        for metric_name, value in episode_metrics.items():
            if value is None:
                continue

            if metric_name not in self.metrics:
                self.metrics[metric_name] = []

            self.metrics[metric_name].append(value)

    def _calculate_avg_score(self, window_size: int = 100) -> float:
        """
        Расчет среднего счета за последние эпизоды

        Args:
            window_size: Размер окна для расчета среднего

        Returns:
            float: Средний счет
        """
        if len(self.scores) >= window_size:
            return sum(self.scores[-window_size:]) / window_size
        else:
            return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def _get_training_time(self) -> str:
        """
        Получение времени обучения в формате ЧЧ:ММ:СС

        Returns:
            str: Время обучения
        """
        if self.start_time is None:
            return "00:00:00"

        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"