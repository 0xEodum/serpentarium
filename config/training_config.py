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

        return True