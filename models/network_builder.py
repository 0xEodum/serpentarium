import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from config.network_config import NetworkConfig


class PyTorchNetworkBuilder:
    """Класс для построения нейронных сетей PyTorch на основе конфигурации"""

    @staticmethod
    def build_network(config: NetworkConfig) -> nn.Module:
        """
        Создает нейронную сеть PyTorch на основе конфигурации

        Args:
            config: Конфигурация сети

        Returns:
            nn.Module: Модель PyTorch
        """

        class DynamicNetwork(nn.Module):
            def __init__(self, input_shape: Tuple[int, ...], layers_config: List[Dict[str, Any]], output_size: int):
                super(DynamicNetwork, self).__init__()

                self.input_shape = input_shape
                self.output_size = output_size

                # Построение слоев сети
                self.layers = nn.ModuleList()
                current_shape = input_shape

                # Флаг для отслеживания был ли уже добавлен flatten
                flatten_added = False

                for layer_config in layers_config:
                    layer_type = layer_config["type"]

                    if layer_type == "conv2d":
                        # Конволюционный слой
                        in_channels = current_shape[0]
                        out_channels = layer_config["filters"]
                        kernel_size = layer_config["kernel_size"]
                        padding = layer_config.get("padding", 0)
                        stride = layer_config.get("stride", 1)

                        self.layers.append(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride
                        ))

                        # Обновляем форму выхода
                        h_out = (current_shape[1] + 2 * padding - kernel_size) // stride + 1
                        w_out = (current_shape[2] + 2 * padding - kernel_size) // stride + 1
                        current_shape = (out_channels, h_out, w_out)

                        # Добавляем активацию, если указана
                        if "activation" in layer_config:
                            self.layers.append(PyTorchNetworkBuilder._get_activation(layer_config["activation"]))

                    elif layer_type == "flatten":
                        # Слой flatten
                        flatten_added = True
                        self.layers.append(nn.Flatten())
                        current_shape = (int(np.prod(current_shape)),)

                    elif layer_type == "dense" or layer_type == "linear":
                        # Если перед dense не было flatten, добавляем его автоматически
                        if len(current_shape) > 1 and not flatten_added:
                            self.layers.append(nn.Flatten())
                            current_shape = (int(np.prod(current_shape)),)
                            flatten_added = True

                        # Полносвязный слой
                        in_features = current_shape[0]
                        out_features = layer_config["units"]

                        self.layers.append(nn.Linear(in_features, out_features))
                        current_shape = (out_features,)

                        # Добавляем активацию, если указана
                        if "activation" in layer_config and layer_config["activation"] != "linear":
                            self.layers.append(PyTorchNetworkBuilder._get_activation(layer_config["activation"]))

                    elif layer_type == "maxpool2d":
                        # Слой макспулинга
                        kernel_size = layer_config["kernel_size"]
                        stride = layer_config.get("stride", kernel_size)
                        padding = layer_config.get("padding", 0)

                        self.layers.append(nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        ))

                        # Обновляем форму выхода
                        h_out = (current_shape[1] + 2 * padding - kernel_size) // stride + 1
                        w_out = (current_shape[2] + 2 * padding - kernel_size) // stride + 1
                        current_shape = (current_shape[0], h_out, w_out)

                    elif layer_type == "dropout":
                        # Слой дропаута
                        p = layer_config.get("rate", 0.5)
                        self.layers.append(nn.Dropout(p=p))

                    else:
                        raise ValueError(f"Неизвестный тип слоя: {layer_type}")

                # Проверяем, что последний слой имеет правильный размер выхода
                if current_shape[0] != output_size:
                    # Если размер не совпадает, добавляем еще один слой
                    self.layers.append(nn.Linear(current_shape[0], output_size))

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

            def _get_conv_out(self, shape):
                # Метод для определения размера выходного тензора после конволюционных слоев
                with torch.no_grad():
                    o = torch.zeros(1, *shape)
                    for layer in self.layers:
                        if isinstance(layer, nn.Flatten):
                            o = layer(o)
                            break
                        o = layer(o)
                    return int(np.prod(o.size()))

        # Создаем сеть на основе конфигурации
        network = DynamicNetwork(
            input_shape=config.input_shape,
            layers_config=config.layers,
            output_size=config.output_size
        )

        return network

    @staticmethod
    def _get_activation(activation_name: str) -> nn.Module:
        """
        Возвращает модуль активации PyTorch по имени

        Args:
            activation_name: Имя функции активации

        Returns:
            nn.Module: Модуль активации
        """
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "softmax": nn.Softmax(dim=1),
            "none": nn.Identity(),
            "linear": nn.Identity()
        }

        if activation_name.lower() not in activations:
            raise ValueError(f"Неизвестная функция активации: {activation_name}")

        return activations[activation_name.lower()]