import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from IPython.display import clear_output


class Visualizer:
    """Класс для визуализации результатов обучения"""

    @staticmethod
    def plot_scores(scores: List[float], avg_scores: List[float],
                    title: str = "Обучение", clear: bool = True,
                    show: bool = True, fig=None, ax=None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Отображение графиков счета

        Args:
            scores: Список со счетом за каждый эпизод
            avg_scores: Список со средним счетом
            title: Заголовок графика
            clear: Очищать ли предыдущий вывод
            show: Отображать ли график
            fig: Существующая фигура (если None, создается новая)
            ax: Существующие оси (если None, создаются новые)

        Returns:
            Tuple[plt.Figure, plt.Axes]: Фигура и оси
        """
        if clear and fig is None:
            clear_output(wait=True)

        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # График счета за каждый эпизод
        ax[0].set_title('Счет за эпизод')
        ax[0].plot(scores)
        ax[0].set_xlabel('Эпизод')
        ax[0].set_ylabel('Счет')

        # График среднего счета
        ax[1].set_title('Средний счет (за 100 эпизодов)')
        ax[1].plot(avg_scores)
        ax[1].set_xlabel('Эпизод')
        ax[1].set_ylabel('Средний счет')

        fig.suptitle(title)
        fig.tight_layout()

        if show and fig is not None:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_metrics(metrics: Dict[str, List[float]],
                     title: str = "Метрики обучения",
                     clear: bool = True,
                     show: bool = True,
                     fig=None, ax=None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Отображение графиков метрик обучения

        Args:
            metrics: Словарь с метриками (ключ - название, значение - список значений)
            title: Заголовок графика
            clear: Очищать ли предыдущий вывод
            show: Отображать ли график
            fig: Существующая фигура (если None, создается новая)
            ax: Существующие оси (если None, создаются новые)

        Returns:
            Tuple[plt.Figure, List[plt.Axes]]: Фигура и оси
        """
        if clear and fig is None:
            clear_output(wait=True)

        # Определяем количество графиков
        n_metrics = len(metrics)

        if n_metrics == 0:
            return fig, ax

        # Определяем размер фигуры
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        if fig is None or ax is None:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

            # Если только одна метрика, то ax не будет массивом
            if n_metrics == 1:
                ax = np.array([ax])

            # Преобразуем в одномерный массив для единообразия обработки
            ax = ax.flatten()

        # Отображаем каждую метрику
        for i, (metric_name, values) in enumerate(metrics.items()):
            if i < len(ax):
                ax[i].set_title(metric_name)
                ax[i].plot(values)
                ax[i].set_xlabel('Шаг')
                ax[i].set_ylabel('Значение')

                # Сглаженная версия для наглядности
                if len(values) > 10:
                    window_size = min(10, len(values) // 10)
                    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                    ax[i].plot(range(window_size - 1, len(values)), smoothed, 'r-', alpha=0.7)

        if fig is not None:
            fig.suptitle(title)
            fig.tight_layout()

            if show:
                plt.show()

        return fig, ax

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