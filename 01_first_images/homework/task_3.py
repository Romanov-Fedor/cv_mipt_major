# type: ignore
from turtle import width
import cv2
import numpy as np
from pyparsing import col

Point = tuple[int, int]


def Rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """

    angle %= 360
    while angle > 90:
        image = Rotate(image, point, 90)
        angle -= 90

    radian_angle = np.deg2rad(angle)
    heigth, width, _ = image.shape
    transition_matrix = np.matrix(
        [[np.sin(radian_angle), np.cos(radian_angle)], [np.cos(radian_angle), np.sin(radian_angle)]])

    vector = np.array([[heigth], [width]])

    new_width, new_heigth = transition_matrix * vector

    new_heigth, new_width = int(np.ceil(new_heigth)), int(np.ceil(new_width))

    first_position = np.float32([[0, 0], [0, heigth], [width, 0]])
    second_position = np.float32([[0, width * np.sin(radian_angle)],
                                  [heigth * np.sin(radian_angle), new_heigth],
                                  [width * np.cos(radian_angle), 0]])

    M = cv2.getAffineTransform(first_position, second_position)

    new_image = cv2.warpAffine(image.copy(), M, (new_width, new_heigth))

    return new_image


def FindNotebookCorners(mask: np.ndarray) -> list[Point]:
    """
    Находит 4 угла тетради

    :param image: маска с тетрадью
    :return: список с четырмя точками: правая верхняя, левая нижняя, правая нижняя, левая верхняя
    """
    def _FindOneCorner(array, start, stop, is_transposed, step=1):

        for index in range(start, stop, step):
            if max(array[index]):
                value_index = int(
                    np.where(array[index] == max(array[index]))[0][0])
                return (value_index, index) if is_transposed else (index, value_index)
    corners = []
    for is_transposed in [False, True]:
        for step in (1, -1):
            array = np.transpose(mask) if is_transposed else mask
            corners.append(_FindOneCorner(array, 0, len(array), is_transposed) if step > 0
                           else _FindOneCorner(array, len(array) - 1, 0, is_transposed, step))
    return corners


def FindNotebookImage(image: np.ndarray) -> np.ndarray:
    """
    С помошью перспективы возвращает изображение, соотетствующее скану

    :param image: картинка, которую сканируем
    :return: отсканированная тетрадь
    """
    red_high, red_low = (6.0, 255, 255), (0.0, 0, 0)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_hsv, red_low, red_high)
    corners = FindNotebookCorners(mask)
    heigth, width, _ = image.shape
    first_position = np.float32([[elem[1], elem[0]] for elem in corners])
    second_position = np.float32(
        [[0, 0], [width, heigth], [0, heigth], [width, 0]])
    M = cv2.getPerspectiveTransform(first_position, second_position)
    new_image = cv2.warpPerspective(image.copy(), M, (width, heigth))
    return new_image


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    # Ваш код
    pass

    return image
