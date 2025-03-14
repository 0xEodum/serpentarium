import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import math
import os
from IPython.display import clear_output

from trainers.base_trainer import BaseTrainer
from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from config.training_config import TrainingConfig
from utils.visualization import Visualizer


class RLTrainer(BaseTrainer):
    """Тренер для алгоритмов обучения с подкреплением"""

    def __init__(self,
                 model: BaseModel,
                 env: BaseEnvironment,
                 config: TrainingConfig):
        """
        Инициализация тренера RL

        Args:
            model: Модель RL для обучения
            env: Окружение
            config: Конфигурация обучения
        """
        super().__init__(model, env, config)

        # Счетчик эпизодов
        self.current_episode = 0

        # Лучший средний счет
        self.best_avg_score = float('-inf')

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Запуск обучения модели RL

        Args:
            num_episodes: Количество эпизодов (если None, используется из конфигурации)

        Returns:
            Dict[str, Any]: Результаты обучения
        """
        num_episodes = num_episodes if num_episodes is not None else self.config.episodes

        # Устанавливаем флаг обучения и время начала
        self.is_training = True
        self.start_time = time.time()

        print(f"Начало обучения на {num_episodes} эпизодов...")

        # Получаем текущий эпсилон (для продолжения обучения)
        epsilon = self._calculate_epsilon(self.current_episode)

        # Определяем, является ли модель PPO по наличию метода add_experience
        is_ppo = hasattr(self.model, 'add_experience')

        # Основной цикл обучения
        for episode in range(num_episodes):
            # Увеличиваем счетчик эпизодов
            self.current_episode += 1

            # Сбрасываем среду и получаем начальное состояние
            state = self.env.reset()
            episode_reward = 0
            episode_metrics = {'game_score': 0, 'total_reward': 0}

            # Цикл эпизода
            while True:
                # Выбираем действие с учетом исследования
                action = self.model.get_action(state, epsilon)

                # Выполняем действие в среде
                next_state, reward, done = self.env.step(action)

                # Для PPO нужна специальная обработка после получения награды
                if is_ppo:
                    self.model.add_experience(reward, done)
                # Для остальных алгоритмов - стандартная обработка
                elif hasattr(self.model, 'memory'):
                    self.model.memory.push(state, action, reward, next_state, done)

                # Обновляем модель
                update_result = self.model.update()

                # Обработка метрик обучения
                for key, value in update_result.items():
                    if value is not None:
                        if key not in episode_metrics:
                            episode_metrics[key] = []
                        episode_metrics[key].append(value)

                # Обновляем текущее состояние и счет
                state = next_state
                episode_reward += reward

                # Если эпизод завершен, выходим из цикла
                if done:
                    break

            # Сохраняем результаты эпизода
            # Для визуализации используем игровой счет (количество съеденной еды)
            game_score = getattr(self.env, 'score', episode_reward)
            self.scores.append(game_score)
            avg_score = self._calculate_avg_score()
            self.avg_scores.append(avg_score)

            # Обновляем финальные метрики эпизода
            episode_metrics['game_score'] = game_score
            episode_metrics['total_reward'] = episode_reward

            # Усредняем накопленные метрики, если они представлены списками
            for key, value in episode_metrics.items():
                if isinstance(value, list) and value:
                    episode_metrics[key] = sum(value) / len(value)

            self._update_metrics(episode_metrics)

            # Рассчитываем новый эпсилон (для алгоритмов, которые его используют)
            epsilon = self._calculate_epsilon(self.current_episode)

            # Сохраняем лучшую модель
            if avg_score > self.best_avg_score:
                self.best_avg_score = avg_score
                best_model_path = os.path.join(
                    os.path.dirname(self.config.model_save_path),
                    f"best_{os.path.basename(self.config.model_save_path)}"
                )
                self.save_model(best_model_path)

            # Отображаем прогресс обучения
            if (episode + 1) % self.config.visualization_freq == 0 or episode == num_episodes - 1:
                clear_output(wait=True)

                # Формируем информационную строку, учитывая тип модели
                if is_ppo:
                    info_str = f"Эпизод: {self.current_episode}/{self.current_episode + num_episodes - episode - 1}, " \
                               f"Счет: {game_score}, Средний счет: {avg_score:.2f}, " \
                               f"Время: {self._get_training_time()}"
                else:
                    info_str = f"Эпизод: {self.current_episode}/{self.current_episode + num_episodes - episode - 1}, " \
                               f"Счет: {game_score}, Средний счет: {avg_score:.2f}, " \
                               f"Epsilon: {epsilon:.4f}, Время: {self._get_training_time()}"

                print(info_str)

                # Визуализируем результаты
                self.visualize_results()

                # Сохраняем модель
                self.save_model()

                # Сохраняем память, если модель поддерживает
                if hasattr(self.model, 'memory') and hasattr(self.model.memory, 'save'):
                    self.model.memory.save(self.config.memory_save_path)

                print(f"Прогресс: {((episode + 1) / num_episodes) * 100:.1f}% "
                      f"({episode + 1}/{num_episodes})")

        # Завершаем обучение
        self.is_training = False
        total_time = self._get_training_time()

        # Возвращаем результаты
        results = {
            'episodes': self.current_episode,
            'final_score': self.scores[-1] if self.scores else None,
            'avg_score': self.avg_scores[-1] if self.avg_scores else None,
            'best_avg_score': self.best_avg_score,
            'training_time': total_time
        }

        print(f"Обучение завершено. Время: {total_time}, "
              f"Лучший средний счет: {self.best_avg_score:.2f}")

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
        eval_rewards = []

        # Определяем, является ли модель PPO
        is_ppo = hasattr(self.model, 'add_experience')

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                # Для PPO не используем эпсилон, но для совместимости интерфейса оставляем epsilon=0
                action = self.model.get_action(state, epsilon=0)

                # Выполняем действие в среде
                next_state, reward, done = self.env.step(action)

                # Если модель PPO, добавляем опыт (для корректного состояния буфера)
                if is_ppo:
                    self.model.add_experience(reward, done)

                # Обновляем текущее состояние и счет
                state = next_state
                episode_reward += reward

                # Если эпизод завершен, выходим из цикла
                if done:
                    break

            # Сохраняем как игровой счет, так и суммарную награду
            game_score = getattr(self.env, 'score', episode_reward)
            eval_scores.append(game_score)
            eval_rewards.append(episode_reward)

            print(f"Эпизод {episode + 1}/{num_episodes}, Счет: {game_score}, Награда: {episode_reward:.2f}")

        avg_score = sum(eval_scores) / len(eval_scores)
        min_score = min(eval_scores)
        max_score = max(eval_scores)

        avg_reward = sum(eval_rewards) / len(eval_rewards)

        print(f"Результаты оценки: Средний счет: {avg_score:.2f}, "
              f"Мин: {min_score}, Макс: {max_score}, "
              f"Средняя награда: {avg_reward:.2f}")

        return {
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'avg_reward': avg_reward,
            'scores': eval_scores,
            'rewards': eval_rewards
        }

    def _calculate_epsilon(self, episode: int) -> float:
        """
        Расчет значения эпсилон для текущего эпизода

        Args:
            episode: Номер текущего эпизода

        Returns:
            float: Значение эпсилон
        """
        # Экспоненциальное убывание
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                  math.exp(-1. * episode / self.config.epsilon_decay)

        return max(self.config.epsilon_end, epsilon)