from typing import Dict, List, Any, Optional
from config.config_base import ConfigBase, ConfigValidationError


class TrainingConfig(ConfigBase):
    """Конфигурация параметров обучения"""

    def __init__(self):
        super().__init__()
        # Общие параметры
        self.model_type = None  # Тип модели: 'dqn', 'double_dqn', 'ppo', 'genetic', etc.
        self.trainer_type = None  # Тип тренера: 'rl', 'genetic', etc.

        # Параметры среды
        self.grid_size = 10  # Размер игрового поля

        # Параметры обучения RL
        self.episodes = 10000  # Количество эпизодов обучения
        self.memory_size = 100000  # Размер буфера воспроизведения
        self.batch_size = 64  # Размер пакета для обучения
        self.gamma = 0.99  # Коэффициент дисконтирования
        self.epsilon_start = 1.0  # Начальное значение epsilon
        self.epsilon_end = 0.01  # Конечное значение epsilon
        self.epsilon_decay = 10000  # Скорость убывания epsilon
        self.target_update = 10  # Частота обновления целевой сети
        self.learning_rate = 0.0001  # Скорость обучения

        # Параметры для PPO
        self.actor_learning_rate = 0.0003  # Скорость обучения для актора
        self.critic_learning_rate = 0.0003  # Скорость обучения для критика
        self.ppo_clip_param = 0.2  # Параметр обрезания для PPO
        self.ppo_epochs = 4  # Количество эпох обучения для одного батча данных
        self.ppo_mini_batches = 4  # Количество мини-батчей
        self.value_loss_coef = 0.5  # Коэффициент для функции потери ценности
        self.entropy_coef = 0.01  # Коэффициент для энтропии
        self.ppo_steps_per_update = 2048  # Количество шагов перед обновлением

        # Параметры для SAC
        self.sac_alpha = 0.2  # Начальное значение коэффициента температуры
        self.sac_auto_alpha = True  # Автоматическая настройка коэффициента температуры
        self.sac_tau = 0.005  # Параметр мягкого обновления целевых сетей
        self.sac_target_update_interval = 1  # Интервал обновления целевых сетей
        self.sac_init_temperature = 0.1  # Начальная температура (когда используется фиксированное значение)

        # Параметры для NEAT и генетических алгоритмов
        self.neat_config_path = "config/neat_config.txt"  # Путь к файлу конфигурации NEAT
        self.num_episodes_per_genome = 3  # Количество эпизодов для оценки одного генома
        self.num_generations = 50  # Общее количество поколений
        self.checkpoint_freq = 10  # Частота сохранения контрольных точек
        self.checkpoint_prefix = "neat-checkpoint-"  # Префикс для файлов контрольных точек

        # Параметры визуализации и сохранения
        self.visualization_freq = 100  # Частота визуализации
        self.model_save_path = "models/trained_model.pth"  # Путь для сохранения модели
        self.memory_save_path = "memory/replay_memory.pkl"  # Путь для сохранения памяти воспроизведения

    def validate(self, expected_type: Optional[str] = None) -> bool:
        """
        Расширенная валидация для конфигурации обучения

        Args:
            expected_type: Ожидаемый тип алгоритма

        Returns:
            bool: Результат валидации

        Raises:
            ConfigValidationError: При неудачной валидации
        """
        # Базовая валидация
        super().validate(expected_type)

        # Проверка наличия обязательных полей
        if self.model_type is None:
            raise ConfigValidationError("Не указан тип модели (model_type)")

        if self.trainer_type is None:
            raise ConfigValidationError("Не указан тип тренера (trainer_type)")

        # Проверка значений параметров
        if self.grid_size <= 0:
            raise ConfigValidationError("Размер игрового поля должен быть положительным числом")

        if self.episodes <= 0:
            raise ConfigValidationError("Количество эпизодов должно быть положительным числом")

        if self.memory_size <= 0:
            raise ConfigValidationError("Размер буфера воспроизведения должен быть положительным числом")

        if self.batch_size <= 0:
            raise ConfigValidationError("Размер пакета должен быть положительным числом")

        if not 0 <= self.gamma <= 1:
            raise ConfigValidationError("Коэффициент дисконтирования должен быть в диапазоне [0, 1]")

        if not 0 <= self.epsilon_start <= 1:
            raise ConfigValidationError("Начальное значение epsilon должно быть в диапазоне [0, 1]")

        if not 0 <= self.epsilon_end <= 1:
            raise ConfigValidationError("Конечное значение epsilon должно быть в диапазоне [0, 1]")

        if self.epsilon_decay <= 0:
            raise ConfigValidationError("Скорость убывания epsilon должна быть положительным числом")

        if self.target_update <= 0:
            raise ConfigValidationError("Частота обновления целевой сети должна быть положительным числом")

        if self.learning_rate <= 0:
            raise ConfigValidationError("Скорость обучения должна быть положительным числом")

        # Проверка параметров PPO, если модель ppo
        if self.model_type == 'ppo':
            if self.actor_learning_rate <= 0:
                raise ConfigValidationError("Скорость обучения актора должна быть положительным числом")

            if self.critic_learning_rate <= 0:
                raise ConfigValidationError("Скорость обучения критика должна быть положительным числом")

            if self.ppo_clip_param <= 0:
                raise ConfigValidationError("Параметр обрезания PPO должен быть положительным числом")

            if self.ppo_epochs <= 0:
                raise ConfigValidationError("Количество эпох обучения PPO должно быть положительным числом")

            if self.ppo_mini_batches <= 0:
                raise ConfigValidationError("Количество мини-батчей PPO должно быть положительным числом")

            if self.ppo_steps_per_update <= 0:
                raise ConfigValidationError("Количество шагов до обновления PPO должно быть положительным числом")

        if self.model_type == 'sac':
            if hasattr(self, 'sac_alpha') and self.sac_alpha <= 0:
                raise ConfigValidationError("Коэффициент температуры SAC (sac_alpha) должен быть положительным числом")

            if hasattr(self, 'sac_tau') and (self.sac_tau <= 0 or self.sac_tau > 1):
                raise ConfigValidationError("Параметр мягкого обновления SAC (sac_tau) должен быть в диапазоне (0, 1]")

            if hasattr(self, 'sac_target_update_interval') and self.sac_target_update_interval <= 0:
                raise ConfigValidationError("Интервал обновления целевых сетей SAC должен быть положительным числом")

            if hasattr(self, 'sac_init_temperature') and self.sac_init_temperature <= 0:
                raise ConfigValidationError("Начальная температура SAC должна быть положительным числом")

            if hasattr(self, 'sac_auto_alpha') and not isinstance(self.sac_auto_alpha, bool):
                raise ConfigValidationError(
                    "Параметр автоматической настройки температуры SAC должен быть булевым значением")

        if self.model_type == 'neat':
            import os

            if not os.path.exists(self.neat_config_path):
                raise ConfigValidationError(f"Файл конфигурации NEAT не найден: {self.neat_config_path}")

            if self.num_episodes_per_genome <= 0:
                raise ConfigValidationError("Количество эпизодов для оценки генома должно быть положительным числом")

            if self.num_generations <= 0:
                raise ConfigValidationError("Количество поколений должно быть положительным числом")

            if self.checkpoint_freq <= 0:
                raise ConfigValidationError("Частота сохранения контрольных точек должна быть положительным числом")

        return True