from typing import Dict, Any, Optional, Type, Union

from environments.base_env import BaseEnvironment
from models.base_model import BaseModel
from trainers.base_trainer import BaseTrainer
from config.network_config import NetworkConfig
from config.training_config import TrainingConfig
from memory.base_memory import BaseMemory


class ModelFactory:
    """Фабрика для создания моделей"""

    # Реестр моделей
    _registry = {}

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Регистрация класса модели

        Args:
            model_type: Тип модели
            model_class: Класс модели
        """
        cls._registry[model_type] = model_class

    @classmethod
    def create(cls,
               model_type: str,
               env: BaseEnvironment,
               network_config: NetworkConfig,
               training_config: TrainingConfig,
               memory: Optional[BaseMemory] = None,
               **kwargs) -> BaseModel:
        """
        Создает модель нужного типа

        Args:
            model_type: Тип модели ('dqn', 'double_dqn', 'ppo', 'genetic', etc.)
            env: Окружение
            network_config: Конфигурация сети
            training_config: Конфигурация обучения
            memory: Объект памяти (опционально)
            **kwargs: Дополнительные аргументы для создания модели

        Returns:
            BaseModel: Созданная модель

        Raises:
            ValueError: Если тип модели не зарегистрирован
        """
        if model_type not in cls._registry:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        model_class = cls._registry[model_type]

        # Проверяем, что конфигурация подходит для данного типа модели
        network_config.validate(expected_type=model_type)

        # Создаем модель
        return model_class(
            env=env,
            network_config=network_config,
            training_config=training_config,
            memory=memory,
            **kwargs
        )


class TrainerFactory:
    """Фабрика для создания тренеров"""

    # Реестр тренеров
    _registry = {}

    @classmethod
    def register(cls, trainer_type: str, trainer_class: Type[BaseTrainer]) -> None:
        """
        Регистрация класса тренера

        Args:
            trainer_type: Тип тренера
            trainer_class: Класс тренера
        """
        cls._registry[trainer_type] = trainer_class

    @classmethod
    def create(cls,
               trainer_type: str,
               model: BaseModel,
               env: BaseEnvironment,
               config: TrainingConfig,
               **kwargs) -> BaseTrainer:
        """
        Создает тренер нужного типа

        Args:
            trainer_type: Тип тренера ('rl', 'genetic', etc.)
            model: Модель для обучения
            env: Окружение
            config: Конфигурация обучения
            **kwargs: Дополнительные аргументы для создания тренера

        Returns:
            BaseTrainer: Созданный тренер

        Raises:
            ValueError: Если тип тренера не зарегистрирован
        """
        if trainer_type not in cls._registry:
            raise ValueError(f"Неизвестный тип тренера: {trainer_type}")

        trainer_class = cls._registry[trainer_type]

        # Создаем тренер
        return trainer_class(
            model=model,
            env=env,
            config=config,
            **kwargs
        )


# Регистрация моделей и тренеров
def register_defaults() -> None:
    """Регистрация стандартных моделей и тренеров"""
    from models.rl.double_dqn import DoubleDQN
    from models.rl.ppo import PPO
    from trainers.rl_trainer import RLTrainer

    # Регистрация моделей
    ModelFactory.register("double_dqn", DoubleDQN)
    ModelFactory.register("ppo", PPO)

    # Регистрация тренеров
    TrainerFactory.register("rl", RLTrainer)


# Регистрируем стандартные модели и тренеры
register_defaults()