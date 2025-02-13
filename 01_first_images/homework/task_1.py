# type: ignore
from turtle import pos, right
import cv2
import numpy as np
from enum import Enum


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

Color = tuple[int, int, int]
Point = tuple[int, int]
Cell = tuple[int, int]


class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    def __neg__(self):
        return Direction(-self.value[0], -self.value[1])


class MazeSolver:
    def __init__(self, image) -> None:
        self.image = image
        self.__blue_image = self._FillBlue()
        self.__direction = Direction.DOWN
        self.__cell_size = len(
            np.where(np.all(image[0] == WHITE, axis=1))[0]) + 2
        self.__height = image.shape[0]

    def _FillBlue(self, point: Point = (0, 0)) -> np.ndarray:
        h, w, _ = self.image.shape
        filled_image = self.image.copy()

        mask = np.zeros((h + 2, w + 2), np.uint8)

        _, filled_image, _, _ = cv2.floodFill(
            filled_image, mask, [point[0], point[1]], BLUE)

        return filled_image

    def _FindEntryAndExit(self) -> tuple[Point, Point]:
        entry = np.where(
            np.all(self.image[0] == WHITE, axis=1))[0]
        exit = np.where(
            np.all(self.image[self.__height - 1] == WHITE, axis=1))[0]

        return ((entry[entry.size // 2], 0), (exit[exit.size // 2], self.__height - 1))

    def _GetCellCenter(self, cell: Cell) -> Point:
        return (cell[0] * (self.__cell_size) + self.__cell_size // 2 + 1, cell[1] * self.__cell_size + self.__cell_size // 2 + 1)

    def _GetWallColor(self, point: Point, direction: Direction) -> Color:
        return tuple(self.__blue_image[point[1] + direction.value[1] * self.__cell_size // 2]
                     [point[0] + direction.value[0] * self.__cell_size // 2])

    def _GetCellWalls(self, cell: Cell) -> dict[Direction, Color | None]:

        cell_center = self._GetCellCenter(cell)
        walls = {}
        for direction in Direction:
            wall_color = self._GetWallColor(cell_center, direction)
            walls[direction] = wall_color if np.any(
                np.not_equal(wall_color, WHITE)) else None
        return walls

    def _GetCellNeighbors(self, cell: Cell) -> dict[Direction, Cell | None]:
        neighbors = {}
        walls = self._GetCellWalls(cell)
        for direction in Direction:
            neighbors[direction] = (cell[0] + direction.value[0], cell[1] +
                                    direction.value[1]) if walls[direction] is None else None
        return neighbors

    def _GetNextPosition(self, cell: Cell, direction: Direction) -> tuple[Cell | None, Direction]:

        def ClockwiseRotation(direction: Direction) -> Direction:
            return Direction(-direction.value[1], direction.value[0])

        def CounterclockwiseRotation(direction: Direction) -> Direction:
            return Direction(direction.value[1], -direction.value[0])

        neighbors = self._GetCellNeighbors(cell)
        walls = self._GetCellWalls(cell)
        bypassing_order = [ClockwiseRotation(
            direction), direction, CounterclockwiseRotation(direction), -direction]

        for possible_direction in bypassing_order:
            if neighbors[possible_direction] is not None:
                neighbors_walls = self._GetCellWalls(
                    neighbors[possible_direction])
                right_wall = neighbors_walls[ClockwiseRotation(
                    possible_direction)]
                left_wall = neighbors_walls[CounterclockwiseRotation(
                    possible_direction)]
                if right_wall == BLUE and left_wall == BLACK:
                    return (neighbors[possible_direction], possible_direction)
                elif right_wall == BLACK:
                    return (neighbors[ClockwiseRotation(possible_direction)], ClockwiseRotation(possible_direction))
                elif left_wall == BLACK:
                    return (neighbors[possible_direction], possible_direction)
            else:
                if walls[possible_direction] == BLACK:
                    return (neighbors[ClockwiseRotation(possible_direction)], ClockwiseRotation(possible_direction))
        return (None, Direction(1, 0))

    def FindWayFromMaze(self) -> tuple:
        """
        Найти путь через лабиринт.

        :param image: изображение лабиринта
        :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
        """
        entry, exit = self._FindEntryAndExit()

        entry_cell = ((entry[0] - self.__cell_size // 2) //
                      self.__cell_size, 0)
        exit_cell = ((exit[0] - self.__cell_size // 2) // self.__cell_size,
                     (exit[1] - self.__cell_size // 2) // self.__cell_size)
        current_cell = entry_cell
        result = ([entry[0], self._GetCellCenter(entry_cell)[0]],
                  [entry[1], self._GetCellCenter(entry_cell)[1]])
        direction = self.__direction
        while current_cell != exit_cell:
            current_cell, direction = self._GetNextPosition(
                current_cell, direction)
            result[0].append(self._GetCellCenter(current_cell)[0])
            result[1].append(self._GetCellCenter(current_cell)[1])
        result[0].append(exit[0])
        result[1].append(exit[1])
        return result
