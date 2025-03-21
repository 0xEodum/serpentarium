import os
import pickle
import random
import numpy as np
import neat
import torch
from typing import Dict, List, Any, Tuple, Optional, Union

from models.base_model import BaseModel
from environments.base_env import BaseEnvironment
from config.neat_config import NEATConfig


class NEAT(BaseModel):
    """Реализация алгоритма NeuroEvolution of Augmenting Topologies (NEAT)"""

    def __init__(self,
                 env: BaseEnvironment,
                 network_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 memory: Optional[Any] = None,  # Не используется в NEAT
                 device: Optional[torch.device] = None):
        """
        Инициализация модели NEAT

        Args:
            env: Окружение
            network_config: Конфигурация нейронной сети (не используется напрямую, нужна для совместимости)
            training_config: Конфигурация обучения
            memory: Не используется в NEAT (для совместимости с интерфейсом)
            device: Устройство для вычислений (для совместимости с интерфейсом)
        """
        self.env = env
        self.training_config = training_config

        # Получаем размеры входных данных и количество действий
        self.input_shape = env.get_state_shape()
        self.action_space = env.get_action_space()
        self.n_actions = self.action_space['n']

        # Определяем количество входных нейронов (уплощаем входную форму)
        self.n_inputs = np.prod(self.input_shape)

        # Создаем конфигурацию NEAT
        self.config_path = training_config.neat_config_path
        self.population = None
        self.best_genome = None
        self.best_net = None

        # Загружаем конфигурацию NEAT
        self._load_neat_config()

        # Инициализируем популяцию
        self.population = neat.Population(self.neat_config)

        # Добавляем репортеры для отслеживания прогресса
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())

        # Счетчик поколений
        self.current_generation = 0

        # Для отслеживания фитнеса
        self.best_fitness = 0.0
        self.current_fitness = 0.0

    def _load_neat_config(self):
        """Загрузка конфигурации NEAT"""
        # Проверяем существование файла
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Файл конфигурации NEAT не найден: {self.config_path}")

        # Определение базовой директории и имени файла
        config_dir = os.path.dirname(self.config_path)
        config_file = os.path.basename(self.config_path)

        # Загружаем конфигурацию
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        # Устанавливаем количество входов и выходов
        self.neat_config.genome_config.num_inputs = self.n_inputs
        self.neat_config.genome_config.num_outputs = self.n_actions

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Выбор действия на основе состояния

        Args:
            state: Текущее состояние
            epsilon: Вероятность случайного действия (для совместимости с интерфейсом)

        Returns:
            int: Выбранное действие
        """
        # Если у нас нет лучшей сети, выбираем случайное действие
        if self.best_net is None:
            return random.randrange(self.n_actions)

        # Преобразуем состояние в подходящий формат (уплощаем)
        flat_state = state.flatten()

        # Получаем выход сети
        output = self.best_net.activate(flat_state)

        # Выбираем действие с максимальным значением
        return np.argmax(output)

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        В контексте NEAT этот метод не имеет прямого аналога,
        так как обновление происходит по окончании оценки всей популяции.

        Returns:
            Dict[str, Any]: Метрики обучения
        """
        return {"fitness": self.current_fitness, "best_fitness": self.best_fitness}

    def save(self, path: str) -> None:
        """
        Сохранение модели

        Args:
            path: Путь для сохранения
        """
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Сохраняем лучший геном и текущую популяцию
        data = {
            'best_genome': self.best_genome,
            'population': self.population,
            'current_generation': self.current_generation,
            'best_fitness': self.best_fitness
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> bool:
        """
        Загрузка модели

        Args:
            path: Путь к файлу модели

        Returns:
            bool: Успех загрузки
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.best_genome = data['best_genome']
            self.population = data['population']
            self.current_generation = data['current_generation']
            self.best_fitness = data.get('best_fitness', 0.0)

            # Воссоздаем сеть из лучшего генома, если он есть
            if self.best_genome:
                self.best_net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.neat_config)

            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели NEAT: {e}")
            return False

    def evolve(self, fitness_function, n_generations=1):
        """
        Эволюционирует популяцию на заданное количество поколений

        Args:
            fitness_function: Функция для оценки фитнеса геномов
            n_generations: Количество поколений для эволюции

        Returns:
            Лучший геном
        """
        for _ in range(n_generations):
            self.current_generation += 1

            # Запускаем один шаг эволюции
            self.best_genome = self.population.run(fitness_function, 1)

            # Обновляем лучшую сеть
            self.best_net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.neat_config)

            # Обновляем лучший фитнес
            self.best_fitness = max(self.best_fitness, self.best_genome.fitness)
            self.current_fitness = self.best_genome.fitness

        return self.best_genome

    def evaluate_genome(self, genome, config, n_episodes=1):
        """
        Оценивает геном на среде

        Args:
            genome: Геном для оценки
            config: Конфигурация NEAT
            n_episodes: Количество эпизодов для оценки

        Returns:
            float: Фитнес генома
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        total_fitness = 0.0

        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_score = 0

            while not done:
                # Преобразуем состояние и получаем действие
                flat_state = state.flatten()
                output = net.activate(flat_state)
                action = np.argmax(output)

                # Выполняем действие
                state, reward, done = self.env.step(action)

                # Накапливаем награду
                episode_score += reward

            # Используем игровой счет (количество съеденной еды) для фитнеса
            game_score = getattr(self.env, 'score', episode_score)
            total_fitness += game_score

        # Средний фитнес по всем эпизодам
        fitness = total_fitness / n_episodes

        return fitness