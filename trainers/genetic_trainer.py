import numpy as np
import time
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import math
from IPython.display import clear_output

from trainers.base_trainer import BaseTrainer
from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from config.training_config import TrainingConfig
from utils.visualization import Visualizer


class GeneticTrainer(BaseTrainer):
    """Тренер для генетических алгоритмов"""

    def __init__(self,
                 model: BaseModel,
                 env: BaseEnvironment,
                 config: TrainingConfig):
        """
        Инициализация тренера генетических алгоритмов

        Args:
            model: Модель (например, NEAT) для обучения
            env: Окружение
            config: Конфигурация обучения
        """
        super().__init__(model, env, config)

        # Счетчик поколений
        self.current_generation = 0

        # Лучший фитнес
        self.best_fitness = float('-inf')

        # Дополнительные метрики
        self.generation_metrics = {}
        self.species_counts = []

    def train(self, num_generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Запуск обучения генетического алгоритма

        Args:
            num_generations: Количество поколений (если None, используется из конфигурации)

        Returns:
            Dict[str, Any]: Результаты обучения
        """
        # Проверяем, что модель поддерживает метод evolve
        if not hasattr(self.model, 'evolve'):
            raise NotImplementedError(
                "Модель не поддерживает метод evolve(), необходимый для генетического обучения"
            )

        num_generations = num_generations if num_generations is not None else self.config.num_generations

        # Устанавливаем флаг обучения и время начала
        self.is_training = True
        self.start_time = time.time()

        print(f"Начало эволюции на {num_generations} поколений...")

        # Создаем функцию фитнеса для оценки геномов
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                fitness = self.model.evaluate_genome(
                    genome,
                    config,
                    n_episodes=self.config.num_episodes_per_genome
                )
                genome.fitness = fitness

                # Обновляем метрики
                if 'fitness_distribution' not in self.generation_metrics:
                    self.generation_metrics['fitness_distribution'] = []
                self.generation_metrics['fitness_distribution'].append(fitness)

        # Основной цикл эволюции
        for generation in range(num_generations):
            # Увеличиваем счетчик поколений
            self.current_generation += 1

            # Сбрасываем метрики для текущего поколения
            self.generation_metrics = {}

            # Запускаем эволюцию на одно поколение
            best_genome = self.model.evolve(eval_genomes, n_generations=1)

            # Получаем метрики
            if hasattr(self.model, 'population') and hasattr(self.model.population, 'species'):
                self.species_counts.append(len(self.model.population.species.species))

            # Собираем статистику
            fitness_values = self.generation_metrics.get('fitness_distribution', [])
            avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
            max_fitness = max(fitness_values) if fitness_values else 0
            min_fitness = min(fitness_values) if fitness_values else 0

            # Обновляем лучший фитнес
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                # Сохраняем лучшую модель
                best_model_path = os.path.join(
                    os.path.dirname(self.config.model_save_path),
                    f"best_{os.path.basename(self.config.model_save_path)}"
                )
                self.save_model(best_model_path)

            # Сохраняем метрики
            self.scores.append(max_fitness)  # Используем максимальный фитнес как счет
            self.avg_scores.append(avg_fitness)

            # Обновляем общие метрики
            metrics_data = {
                'max_fitness': max_fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'species_count': self.species_counts[-1] if self.species_counts else 0
            }
            self._update_metrics(metrics_data)

            # Отображаем прогресс обучения
            if (generation + 1) % self.config.visualization_freq == 0 or generation == num_generations - 1:
                clear_output(wait=True)

                # Формируем информационную строку
                info_str = f"Поколение: {self.current_generation}/{self.current_generation + num_generations - generation - 1}, " \
                           f"Лучший фитнес: {max_fitness:.2f}, Средний фитнес: {avg_fitness:.2f}, " \
                           f"Количество видов: {self.species_counts[-1] if self.species_counts else 0}, " \
                           f"Время: {self._get_training_time()}"

                print(info_str)

                # Визуализируем результаты
                self.visualize_results()

                # Сохраняем модель
                self.save_model()

                print(f"Прогресс: {((generation + 1) / num_generations) * 100:.1f}% "
                      f"({generation + 1}/{num_generations})")

            # Сохраняем контрольную точку, если нужно
            if hasattr(self.config, 'checkpoint_freq') and \
                    hasattr(self.config, 'checkpoint_prefix') and \
                    (generation + 1) % self.config.checkpoint_freq == 0:

                checkpoint_path = f"{self.config.checkpoint_prefix}{self.current_generation}"
                if hasattr(self.model.population, 'save_checkpoint'):
                    self.model.population.save_checkpoint(checkpoint_path)
                    print(f"Сохранена контрольная точка: {checkpoint_path}")

        # Завершаем обучение
        self.is_training = False
        total_time = self._get_training_time()

        # Возвращаем результаты
        results = {
            'generations': self.current_generation,
            'final_fitness': self.scores[-1] if self.scores else None,
            'avg_fitness': self.avg_scores[-1] if self.avg_scores else None,
            'best_fitness': self.best_fitness,
            'species_count': self.species_counts[-1] if self.species_counts else 0,
            'training_time': total_time
        }

        print(f"Эволюция завершена. Время: {total_time}, "
              f"Лучший фитнес: {self.best_fitness:.2f}")

        return results

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Оценка модели

        Args:
            num_episodes: Количество эпизодов для оценки

        Returns:
            Dict[str, Any]: Результаты оценки
        """
        print(f"Оценка модели на {num_episodes} эпизодах...")

        eval_scores = []

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Выбираем действие
                action = self.model.get_action(state)

                # Выполняем действие в среде
                next_state, reward, done = self.env.step(action)

                # Обновляем текущее состояние и счет
                state = next_state
                episode_reward += reward

            # Сохраняем игровой счет
            game_score = getattr(self.env, 'score', episode_reward)
            eval_scores.append(game_score)

            print(f"Эпизод {episode + 1}/{num_episodes}, Счет: {game_score}")

        avg_score = sum(eval_scores) / len(eval_scores)
        min_score = min(eval_scores)
        max_score = max(eval_scores)

        print(f"Результаты оценки: Средний счет: {avg_score:.2f}, "
              f"Мин: {min_score}, Макс: {max_score}")

        return {
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'scores': eval_scores
        }

    def visualize_results(self) -> None:
        """
        Визуализация результатов обучения

        Расширяет базовую визуализацию метриками, специфичными для генетических алгоритмов
        """
        from utils.visualization import Visualizer

        # Очищаем вывод перед отображением всех графиков
        from IPython.display import clear_output
        clear_output(wait=True)

        # Определяем количество строк для метрик
        n_metrics = len(self.metrics) if self.metrics else 0
        n_metric_rows = (n_metrics + 2) // 3  # Используем до 3 столбцов для метрик

        # Определяем общее количество строк: 1 для счетов + строки для метрик + 1 для видов
        n_rows = 1 + (n_metric_rows if n_metrics > 0 else 0) + (1 if self.species_counts else 0)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 6 * n_rows))

        # Создаем оси для графиков счета (первая строка, 2 столбца)
        if self.scores:
            ax_scores = plt.subplot2grid((n_rows, 3), (0, 0), colspan=1)
            ax_avg_scores = plt.subplot2grid((n_rows, 3), (0, 1), colspan=1)

            # Если есть данные о видах, добавляем третий график
            ax_species = None
            if self.species_counts:
                ax_species = plt.subplot2grid((n_rows, 3), (0, 2), colspan=1)

            # График максимального фитнеса за поколение
            ax_scores.set_title('Максимальный фитнес за поколение')
            ax_scores.plot(self.scores)
            ax_scores.set_xlabel('Поколение')
            ax_scores.set_ylabel('Фитнес')

            # График среднего фитнеса
            ax_avg_scores.set_title('Средний фитнес за поколение')
            ax_avg_scores.plot(self.avg_scores)
            ax_avg_scores.set_xlabel('Поколение')
            ax_avg_scores.set_ylabel('Средний фитнес')

            # График количества видов
            if ax_species is not None and self.species_counts:
                ax_species.set_title('Количество видов')
                ax_species.plot(self.species_counts)
                ax_species.set_xlabel('Поколение')
                ax_species.set_ylabel('Количество видов')

        # Создаем графики для метрик (начиная со второй строки)
        if self.metrics:
            for i, (metric_name, values) in enumerate(self.metrics.items()):
                if metric_name == 'species_count':
                    continue  # Уже отображено выше

                row = 1 + i // 3
                col = i % 3
                ax = plt.subplot2grid((n_rows, 3), (row, col))

                ax.set_title(metric_name)
                ax.plot(values)
                ax.set_xlabel('Поколение')
                ax.set_ylabel('Значение')

                # Сглаженная версия для наглядности
                if len(values) > 10:
                    window_size = min(10, len(values) // 10)
                    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                    ax.plot(range(window_size - 1, len(values)), smoothed, 'r-', alpha=0.7)

        plt.tight_layout()
        plt.show()