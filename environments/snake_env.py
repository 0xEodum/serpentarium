import numpy as np
import random
from typing import Tuple, Dict, Any, List, Union, Optional
from environments.base_env import BaseEnvironment


class SnakeEnvironment(BaseEnvironment):
    """Реализация среды игры 'Змейка'"""

    def __init__(self, grid_size: int = 10):
        """
        Инициализация среды

        Args:
            grid_size: Размер игрового поля
        """
        self.grid_size = grid_size
        self.snake = []
        self.direction = None
        self.food = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Сброс среды и возврат начального состояния

        Returns:
            np.ndarray: Начальное состояние
        """
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = 1  # Начинаем движение вправо
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.place_food()
        return self.get_state()

    def place_food(self) -> None:
        """Размещение еды на игровом поле"""
        while True:
            self.food = (random.randint(0, self.grid_size - 1),
                         random.randint(0, self.grid_size - 1))
            if self.food not in self.snake:
                break

    def get_state(self) -> np.ndarray:
        """
        Получение текущего состояния

        Returns:
            np.ndarray: Текущее состояние, представленное в виде
                        матрицы с 3 каналами: тело змейки, голова, еда
        """
        # Создаем карту в виде матрицы с 3 каналами: змейка, голова, еда
        state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Канал 0: тело змейки
        for segment in self.snake:
            state[0, segment[1], segment[0]] = 1

        # Канал 1: голова змейки (отдельно)
        head = self.snake[0]
        state[1, head[1], head[0]] = 1

        # Канал 2: еда
        state[2, self.food[1], self.food[0]] = 1

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Выполнение действия

        Args:
            action: Выбранное действие
                   0: вверх, 1: вправо, 2: вниз, 3: влево

        Returns:
            Tuple[np.ndarray, float, bool]:
                - Новое состояние
                - Награда
                - Флаг завершения эпизода
        """
        # Предотвращаем поворот на 180 градусов
        if (action == 0 and self.direction == 2) or \
                (action == 2 and self.direction == 0) or \
                (action == 1 and self.direction == 3) or \
                (action == 3 and self.direction == 1):
            action = self.direction

        self.direction = action

        # Обновляем координаты головы змейки
        head_x, head_y = self.snake[0]
        if action == 0:  # вверх
            head_y -= 1
        elif action == 1:  # вправо
            head_x += 1
        elif action == 2:  # вниз
            head_y += 1
        else:  # влево
            head_x -= 1

        # Проверяем столкновение со стеной
        if head_x < 0 or head_x >= self.grid_size or \
                head_y < 0 or head_y >= self.grid_size:
            self.game_over = True
            return self.get_state(), -10, True

        # Проверяем столкновение с телом змейки
        if (head_x, head_y) in self.snake:
            self.game_over = True
            return self.get_state(), -10, True

        # Добавляем новую голову
        self.snake.insert(0, (head_x, head_y))

        # Проверяем, съела ли змейка еду
        reward = 0
        if (head_x, head_y) == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            # Если не съела, удаляем последний сегмент
            self.snake.pop()

            # Небольшое отрицательное вознаграждение за каждый шаг
            reward = -0.01

            # Бонус за приближение к еде
            prev_head = self.snake[1] if len(self.snake) > 1 else self.snake[0]
            prev_distance = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])
            curr_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

            if curr_distance < prev_distance:
                reward += 0.1

        # Увеличиваем счетчик шагов
        self.steps += 1

        # Ограничиваем количество шагов
        if self.steps > 100 * self.grid_size and self.score < 5:
            self.game_over = True
            return self.get_state(), -10, True

        return self.get_state(), reward, self.game_over

    def get_state_shape(self) -> Tuple[int, ...]:
        """
        Возвращает форму состояния для инициализации сети

        Returns:
            Tuple[int, ...]: Форма состояния (каналы, высота, ширина)
        """
        return (3, self.grid_size, self.grid_size)

    def get_action_space(self) -> Dict[str, Any]:
        """
        Возвращает информацию о пространстве действий

        Returns:
            Dict[str, Any]: Информация о пространстве действий
                - 'n': Количество действий (4)
                - 'actions': Словарь с описанием действий
        """
        return {
            'n': 4,
            'actions': {
                0: 'вверх',
                1: 'вправо',
                2: 'вниз',
                3: 'влево'
            }
        }

    def render(self, mode: str = 'text') -> Optional[str]:
        """
        Отображение текущего состояния среды

        Args:
            mode: Режим отображения ('text' для текстового представления)

        Returns:
            Optional[str]: Текстовое представление игрового поля (если mode='text')
        """
        if mode == 'text':
            grid = [['·' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

            # Отображаем тело змейки
            for i, segment in enumerate(self.snake):
                if i == 0:
                    # Голова
                    if self.direction == 0:  # вверх
                        grid[segment[1]][segment[0]] = '▲'
                    elif self.direction == 1:  # вправо
                        grid[segment[1]][segment[0]] = '►'
                    elif self.direction == 2:  # вниз
                        grid[segment[1]][segment[0]] = '▼'
                    else:  # влево
                        grid[segment[1]][segment[0]] = '◄'
                else:
                    # Тело
                    grid[segment[1]][segment[0]] = '■'

            # Отображаем еду
            grid[self.food[1]][self.food[0]] = '×'

            # Формируем текстовое представление
            text_repr = f"Счет: {self.score}, Шаги: {self.steps}\n"
            text_repr += '\n'.join([''.join(row) for row in grid])

            return text_repr

        return None