import yaml
import json
import os
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, ClassVar

T = TypeVar('T', bound='ConfigBase')


class ConfigValidationError(Exception):
    """Исключение, возникающее при ошибках валидации конфигурации"""
    pass


class ConfigBase:
    """Базовый класс для всех конфигураций"""

    def __init__(self):
        self.metadata = {
            "type": None,
            "version": "1.0",
            "compatible_with": []
        }

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация конфигурации в словарь"""
        result = {"metadata": self.metadata}

        # Добавляем все атрибуты, кроме служебных (начинающихся с _)
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'metadata':
                result[key] = value

        return result

    def from_dict(self, config_dict: Dict[str, Any]) -> 'ConfigBase':
        """
        Загрузка конфигурации из словаря

        Args:
            config_dict: Словарь с конфигурацией

        Returns:
            self: Текущий экземпляр для цепочки вызовов
        """
        if 'metadata' in config_dict:
            self.metadata = config_dict['metadata']

        # Загружаем все остальные атрибуты
        for key, value in config_dict.items():
            if key != 'metadata' and hasattr(self, key):
                setattr(self, key, value)

        return self

    def save(self, file_path: str) -> None:
        """
        Сохранение конфигурации в файл

        Args:
            file_path: Путь для сохранения (.yaml или .json)
        """
        config_dict = self.to_dict()

        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Определяем формат по расширению файла
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Поддерживаются только форматы .yaml, .yml или .json")

    @classmethod
    def load(cls: Type[T], file_path: str) -> T:
        """
        Загрузка конфигурации из файла

        Args:
            file_path: Путь к файлу конфигурации

        Returns:
            Загруженный экземпляр конфигурации
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {file_path}")

        # Определяем формат по расширению файла
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Поддерживаются только форматы .yaml, .yml или .json")

        # Создаем и заполняем экземпляр
        instance = cls()
        instance.from_dict(config_dict)

        return instance

    def validate(self, expected_type: Optional[str] = None) -> bool:
        """
        Валидирует конфигурацию

        Args:
            expected_type: Ожидаемый тип алгоритма

        Returns:
            bool: Результат валидации

        Raises:
            ConfigValidationError: При неудачной валидации
        """
        # Проверка совместимости типа алгоритма
        if expected_type and expected_type not in [self.metadata["type"]] + self.metadata["compatible_with"]:
            raise ConfigValidationError(
                f"Несовместимый тип алгоритма. Ожидается {expected_type}, "
                f"но конфигурация предназначена для {self.metadata['type']} "
                f"(совместима с {', '.join(self.metadata['compatible_with'])})"
            )

        return True


class ConfigValidator:
    """Базовый класс для валидации конфигураций"""

    @staticmethod
    def validate(config: ConfigBase, expected_type: Optional[str] = None) -> bool:
        """
        Проверяет валидность конфигурации

        Args:
            config: Объект конфигурации для проверки
            expected_type: Ожидаемый тип алгоритма (если указан)

        Returns:
            bool: Результат валидации

        Raises:
            ConfigValidationError: При обнаружении проблем с конфигурацией
        """
        return config.validate(expected_type)