import os
from typing import Dict, List, Any, Optional
from config.config_base import ConfigBase, ConfigValidationError


class NEATConfig(ConfigBase):
    """Конфигурация для алгоритма NEAT"""

    def __init__(self):
        super().__init__()

        # Устанавливаем тип алгоритма
        self.metadata = {
            "type": "neat",
            "version": "1.0",
            "compatible_with": ["genetic"]
        }

        # Параметры популяции
        self.population_size = 50
        self.fitness_threshold = 100

        # Параметры мутации
        self.weight_mutate_rate = 0.8
        self.node_add_prob = 0.2
        self.node_delete_prob = 0.2
        self.conn_add_prob = 0.5
        self.conn_delete_prob = 0.5

        # Параметры видов
        self.compatibility_threshold = 3.0
        self.species_elitism = 2
        self.max_stagnation = 20

        # Параметры репродукции
        self.elitism = 2
        self.survival_threshold = 0.2

        # Путь к конфигурационному файлу
        self.config_path = "config/neat_config.txt"

        # Параметры оценки
        self.num_episodes_per_genome = 3  # Количество эпизодов для оценки одного генома
        self.num_generations = 100  # Общее количество поколений

        # Дополнительные параметры
        self.checkpoint_freq = 10  # Частота сохранения контрольных точек
        self.checkpoint_prefix = "neat-checkpoint-"  # Префикс для файлов контрольных точек

    def validate(self, expected_type: Optional[str] = None) -> bool:
        """
        Расширенная валидация для конфигурации NEAT

        Args:
            expected_type: Ожидаемый тип алгоритма

        Returns:
            bool: Результат валидации

        Raises:
            ConfigValidationError: При неудачной валидации
        """
        # Базовая валидация
        super().validate(expected_type)

        # Проверка существования файла конфигурации
        if not os.path.exists(self.config_path):
            raise ConfigValidationError(f"Файл конфигурации NEAT не найден: {self.config_path}")

        # Проверка значений параметров
        if self.population_size <= 0:
            raise ConfigValidationError("Размер популяции должен быть положительным числом")

        if self.num_episodes_per_genome <= 0:
            raise ConfigValidationError("Количество эпизодов для оценки генома должно быть положительным числом")

        if self.num_generations <= 0:
            raise ConfigValidationError("Количество поколений должно быть положительным числом")

        if not 0 <= self.weight_mutate_rate <= 1:
            raise ConfigValidationError("Скорость мутации весов должна быть в диапазоне [0, 1]")

        if not 0 <= self.node_add_prob <= 1:
            raise ConfigValidationError("Вероятность добавления узла должна быть в диапазоне [0, 1]")

        if not 0 <= self.node_delete_prob <= 1:
            raise ConfigValidationError("Вероятность удаления узла должна быть в диапазоне [0, 1]")

        if not 0 <= self.conn_add_prob <= 1:
            raise ConfigValidationError("Вероятность добавления связи должна быть в диапазоне [0, 1]")

        if not 0 <= self.conn_delete_prob <= 1:
            raise ConfigValidationError("Вероятность удаления связи должна быть в диапазоне [0, 1]")

        if self.compatibility_threshold <= 0:
            raise ConfigValidationError("Порог совместимости должен быть положительным числом")

        if self.species_elitism < 0:
            raise ConfigValidationError("Элитизм видов должен быть неотрицательным числом")

        if self.max_stagnation <= 0:
            raise ConfigValidationError("Максимальная стагнация должна быть положительным числом")

        if self.elitism < 0:
            raise ConfigValidationError("Элитизм должен быть неотрицательным числом")

        if not 0 <= self.survival_threshold <= 1:
            raise ConfigValidationError("Порог выживания должен быть в диапазоне [0, 1]")

        return True

    def update_neat_config_file(self, input_dims=None, output_dims=None):
        """
        Обновляет файл конфигурации NEAT на основе текущих параметров

        Args:
            input_dims (int, optional): Количество входных нейронов
            output_dims (int, optional): Количество выходных нейронов
        """
        # Определяем значения входов/выходов, если не указаны
        num_inputs = input_dims if input_dims is not None else 3 * 10 * 10  # По умолчанию для Snake 10x10
        num_outputs = output_dims if output_dims is not None else 4  # По умолчанию 4 действия

        # Содержимое конфигурационного файла
        config_content = f"""# NEAT конфигурация для игры Змейка

[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.fitness_threshold}
pop_size              = {self.population_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = {self.conn_add_prob}
conn_delete_prob        = {self.conn_delete_prob}

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = {self.weight_mutate_rate}
weight_replace_rate     = 0.1

# network parameters
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}

# node add/remove rates
node_add_prob           = {self.node_add_prob}
node_delete_prob        = {self.node_delete_prob}

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# network parameters
feed_forward            = True
initial_connection      = full

[DefaultSpeciesSet]
compatibility_threshold = {self.compatibility_threshold}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {self.max_stagnation}
species_elitism      = {self.species_elitism}

[DefaultReproduction]
elitism            = {self.elitism}
survival_threshold = {self.survival_threshold}
"""

        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
        # Записываем содержимое в файл
        with open(self.config_path, 'w') as f:
            f.write(config_content)