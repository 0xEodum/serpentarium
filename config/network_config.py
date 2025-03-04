from typing import Dict, List, Any, Tuple, Optional, Union
import torch.nn as nn
from config.config_base import ConfigBase, ConfigValidationError


class NetworkConfigValidator:
    """Валидатор для конфигурации нейросети"""

    @staticmethod
    def validate_layer_transitions(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Проверяет переходы между слоями и автоматически добавляет
        необходимые промежуточные слои (например, flatten)

        Args:
            layers: Список слоев

        Returns:
            Список слоев с добавленными промежуточными слоями
        """
        if not layers:
            return layers

        adjusted_layers = []
        need_flatten = False

        for i, layer in enumerate(layers):
            if i > 0:
                # Если переход от conv к dense без flatten между ними
                prev_layer = layers[i - 1]
                if prev_layer["type"].startswith("conv") and layer["type"] == "dense" and not need_flatten:
                    adjusted_layers.append({"type": "flatten"})
                    need_flatten = True

            adjusted_layers.append(layer)

            # Сброс флага после добавления flatten
            if layer["type"] == "flatten":
                need_flatten = False

        return adjusted_layers

    @staticmethod
    def validate_dimensions(input_shape: Tuple[int, ...], layers: List[Dict[str, Any]]) -> bool:
        """
        Проверяет корректность размерностей между слоями

        Args:
            input_shape: Форма входных данных
            layers: Список слоев

        Returns:
            bool: Результат валидации

        Raises:
            ConfigValidationError: При обнаружении проблем с размерностями
        """
        # TODO: Реализовать проверку размерностей для различных типов слоев
        return True


class NetworkConfig(ConfigBase):
    """Конфигурация архитектуры нейронной сети"""

    def __init__(self):
        super().__init__()
        self.input_shape = None  # Форма входных данных (каналы, высота, ширина)
        self.output_size = None  # Размер выходного слоя (количество действий)
        self.layers = []  # Список слоев

    def validate(self, expected_type: Optional[str] = None) -> bool:
        """
        Расширенная валидация для сетевой конфигурации

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
        if self.input_shape is None:
            raise ConfigValidationError("Не указана форма входных данных (input_shape)")

        if self.output_size is None:
            raise ConfigValidationError("Не указан размер выходного слоя (output_size)")

        if not self.layers:
            raise ConfigValidationError("Не указаны слои нейронной сети (layers)")

        # Проверка наличия нужных слоев и автоматическое добавление промежуточных слоев
        self.layers = NetworkConfigValidator.validate_layer_transitions(self.layers)

        # Проверка размерностей
        NetworkConfigValidator.validate_dimensions(self.input_shape, self.layers)

        return True

    def create_network(self, framework: str = "pytorch") -> nn.Module:
        """
        Создает нейронную сеть на основе конфигурации

        Args:
            framework: Фреймворк для создания сети ("pytorch" или "tensorflow")

        Returns:
            Объект нейронной сети
        """
        if framework != "pytorch":
            raise NotImplementedError(f"Фреймворк {framework} пока не поддерживается")

        from serpentarium.models.network_builder import PyTorchNetworkBuilder
        return PyTorchNetworkBuilder.build_network(self)