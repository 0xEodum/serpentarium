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

        # Очищаем вывод перед отображением всех графиков
        from IPython.display import clear_output
        clear_output(wait=True)

        # Создаем подграфики
        # Определяем количество строк для метрик
        n_metrics = len(self.metrics) if self.metrics else 0
        n_metric_rows = (n_metrics + 2) // 3  # Используем до 3 столбцов для метрик

        # Определяем общее количество строк: 1 для счетов + строки для метрик
        n_rows = 1 + (n_metric_rows if n_metrics > 0 else 0)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 6 * n_rows))

        # Создаем оси для графиков счета (первая строка, 2 столбца)
        if self.scores:
            ax_scores = plt.subplot2grid((n_rows, 3), (0, 0), colspan=1)
            ax_avg_scores = plt.subplot2grid((n_rows, 3), (0, 1), colspan=1)

            # График счета за эпизод (игровой счет - количество съеденной еды)
            ax_scores.set_title('Игровой счет за эпизод')
            ax_scores.plot(self.scores)
            ax_scores.set_xlabel('Эпизод')
            ax_scores.set_ylabel('Счет')

            # График среднего счета
            ax_avg_scores.set_title('Средний игровой счет (за 100 эпизодов)')
            ax_avg_scores.plot(self.avg_scores)
            ax_avg_scores.set_xlabel('Эпизод')
            ax_avg_scores.set_ylabel('Средний счет')

        # Создаем графики для метрик (начиная со второй строки)
        if self.metrics:
            for i, (metric_name, values) in enumerate(self.metrics.items()):
                row = 1 + i // 3
                col = i % 3
                ax = plt.subplot2grid((n_rows, 3), (row, col))

                ax.set_title(metric_name)
                ax.plot(values)
                ax.set_xlabel('Шаг')
                ax.set_ylabel('Значение')

                # Сглаженная версия для наглядности
                if len(values) > 10:
                    window_size = min(10, len(values) // 10)
                    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                    ax.plot(range(window_size - 1, len(values)), smoothed, 'r-', alpha=0.7)

        plt.tight_layout()
        plt.show()

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