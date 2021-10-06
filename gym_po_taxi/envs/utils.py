__all__ = ['TAXI_ROOMS_LAYOUT', 'BASE_SHAPE', 'ACTIONS_BASE', 'ACTIONS_HARD', 'VIEW_3x3', 'LOCATIONS', 'FLAT_ACTIONS_BASE', 'FORWARD_ACTIONS']

import numpy as np
from enum import Enum, IntEnum

"""This file contains common Taxi maps

Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
    The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination
    (another one of the four specified locations), and then drops off the passenger.
    Once the passenger is dropped off, the episode ends.
"""

"""Base map from original. Not used"""
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

"""Walkability map. 0 is invalid, 1 and above are valid

2 is red, 3, is yellow, 4 is blue, 5 is green
"""
TAXI_ROOMS_LAYOUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 0, 1, 1, 5, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 3, 0, 1, 0, 4, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int).T

BASE_SHAPE = TAXI_ROOMS_LAYOUT.shape
VIEW_3x3 = np.array([0, -9, -8, -7, -1, 1, 7, 8, 9], dtype=int)  # 3x3 grid around agent, flattened, middle index first

"""Red, Yellow, Blue, Green locations"""
LOCATIONS = {
    'R': np.array([1, 1]),
    'Y': np.array([1, 6]),
    'B': np.array([5, 6]),
    'G': np.array([6, 1])
}

"""w.r.t. base shape"""
FLAT_ACTIONS_BASE = np.array([
    0,  # No-op
    -1,  # Up
    1,  # Down
    -8,  #Left
    8,  # Right
    0  # Pickup/dropoff
], dtype=int)

FORWARD_ACTIONS = np.array([
    -1,   # Up
    8,  # Right
    1,  # Down
    -8  # Left
], dtype=int)

ACTIONS_BASE = {
    0: np.array([0, 0]),  # No-op
    1: np.array([0, -1]),  # Up
    2: np.array([0, 1]),  # Down
    3: np.array([1, 0]),  # Right
    4: np.array([-1, 0]),  # Left
    5: np.array([0, 0]),  # Pickup
}

ACTIONS_HARD = {
    0: np.array([0, 1]),  # No-op
    1: np.array([0, -1]),  # Turn
    2: np.array([1, 0]),
    3: np.array([-1, 0]),
}