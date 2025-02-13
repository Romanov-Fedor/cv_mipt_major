from pyclbr import Class
import cv2
import numpy as np


def GetRoadsPositions(road_image: np.ndarray) -> list[tuple[int, int]]:
    """
    Получить координаты дорог

    :param road_image: изображение с выделенными дорогами
    :return: список с координатами по x начала и конца дорог
    """
    roads_position = []
    road_len = 0
    for i in range(len(road_image[0])):
        if road_image[0][i]:
            road_len += 1
        if road_image[0][i] == 0 and road_len != 0:
            roads_position.append((i - road_len, i))
            road_len = 0
    return roads_position


def GetCarPosition(car_image: np.ndarray) -> tuple[int, int]:
    """
    Получить позицию какого-то пиксель автомобиля

    :param car_image: изображение с выделенной машиной
    :return: координаты машины (x, y)
    """

    for i in range(len(car_image)):
        for j in range(len(car_image[i])):
            if car_image[i][j]:
                return (i, j)
    return (-1, -1)


def GetObstaclePositions(obstacle_image: np.ndarray, roads_positions: list[tuple[int, int]]) -> list[int]:
    """
    Получить номера дорог, на которых находятся препятствия

    :param obstacle_image: изображение с выделеннымми препятствиями
    :param roads_positions: координаты начала и конца дорог

    :return: список с номерами дорог, на которых находятся препятствия 
    """

    obstacle_positions = []
    for road in roads_positions:
        necessary_pixel = road[1] - (road[1] - road[0]) // 2
        for i in range(len(obstacle_image)):
            if obstacle_image[i][necessary_pixel]:
                obstacle_positions.append(roads_positions.index(road))
                break

    return obstacle_positions


def FindRoadNumber(image: np.ndarray) -> int | str:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """

    grey_high, grey_low = (180.0, 18, 230), (0.0, 0, 40)
    blue_high, blue_low = (128.0, 255, 255), (90.0, 50, 70)
    red_high, red_low = (9.0, 255, 255), (0.0, 50, 70)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    car_image = cv2.inRange(image_hsv, blue_low, blue_high)  # type: ignore
    roads_image = cv2.inRange(image_hsv, grey_low, grey_high)  # type: ignore
    obstacle_image = cv2.inRange(image_hsv, red_low, red_high)  # type: ignore
    roads_positions = GetRoadsPositions(roads_image)
    car_position = GetCarPosition(car_image)
    obstacle_positions = GetObstaclePositions(obstacle_image, roads_positions)
    car_road = -1
    for i in range(len(roads_positions)):
        if roads_positions[i][0] <= car_position[1] <= roads_positions[i][1]:
            car_road = i

    for road_number in range(len(roads_positions)):
        if road_number not in obstacle_positions:
            # if car_road == road_number:
            #     return "There is no need to change lanes. The road is clear"
            # else:
            return road_number

    return -1
