import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from IPython.display import clear_output


class Visualizer:
    """Класс для визуализации результатов обучения"""

    @staticmethod
    def plot_scores(scores: List[float], avg_scores: List[float],
                    title: str = "Обучение", clear: bool = True,
                    show: bool = True) -> None:
        """
        Отображение графиков счета

        Args:
            scores: Список со счетом за каждый эпизод
            avg_scores: Список со средним счетом
            title: Заголовок графика
            clear: Очищать ли предыдущий вывод
            show: Отображать ли график
        """
        if clear:
            clear_output(wait=True)

        plt.figure(figsize=(12, 5))

        # График счета за каждый эпизод
        plt.subplot(1, 2, 1)
        plt.title('Счет за эпизод')
        plt.plot(scores)
        plt.xlabel('Эпизод')
        plt.ylabel('Счет')

        # График среднего счета
        plt.subplot(1, 2, 2)
        plt.title('Средний счет (за 100 эпизодов)')
        plt.plot(avg_scores)
        plt.xlabel('Эпизод')
        plt.ylabel('Средний счет')

        plt.suptitle(title)
        plt.tight_layout()

        if show:
            plt.show()

    @staticmethod
    def plot_metrics(metrics: Dict[str, List[float]],
                     title: str = "Метрики обучения",
                     clear: bool = True,
                     show: bool = True) -> None:
        """
        Отображение графиков метрик обучения

        Args:
            metrics: Словарь с метриками (ключ - название, значение - список значений)
            title: Заголовок графика
            clear: Очищать ли предыдущий вывод
            show: Отображать ли график
        """
        if clear:
            clear_output(wait=True)

        # Определяем количество графиков
        n_metrics = len(metrics)

        if n_metrics == 0:
            return

        # Определяем размер фигуры
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        plt.figure(figsize=(6 * n_cols, 4 * n_rows))

        # Отображаем каждую метрику
        for i, (metric_name, values) in enumerate(metrics.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            plt.title(metric_name)
            plt.plot(values)
            plt.xlabel('Шаг')
            plt.ylabel('Значение')

            # Сглаженная версия для наглядности
            if len(values) > 10:
                window_size = min(10, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size - 1, len(values)), smoothed, 'r-', alpha=0.7)

        plt.suptitle(title)
        plt.tight_layout()

        if show:
            plt.show()

    @staticmethod
    def visualize_state(state: np.ndarray, title: str = "Состояние") -> None:
        """
        Визуализация состояния среды

        Args:
            state: Состояние для визуализации
            title: Заголовок графика
        """
        if len(state.shape) != 3:
            raise ValueError("Ожидается 3D-состояние (каналы, высота, ширина)")

        n_channels = state.shape[0]
        plt.figure(figsize=(4 * n_channels, 4))

        for i in range(n_channels):
            plt.subplot(1, n_channels, i + 1)
            plt.imshow(state[i], cmap='viridis')
            plt.title(f"Канал {i}")
            plt.colorbar()

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()