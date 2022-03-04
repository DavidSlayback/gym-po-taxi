import numpy as np
from brax import jumpy as jp

# Qpos and others
POSITION = {
    'acrobot': jp.arange(0, 2),
    'ant': jp.arange(0, 13),
    'fetch': jp.concatenate((jp.arange(0, 6), jp.arange(10, 49)), axis=0),
    'grasp': jp.arange(8, 56),
    'halfcheetah': jp.arange(0, 11),
    'hopper': jp.arange(0, 8),
    'humanoid': jp.concatenate((jp.arange(0, 22), jp.arange(45, 144)), axis=0),
    'humanoidstandup': jp.concatenate((jp.arange(0, 22), jp.arange(45, 144)), axis=0),
    'inverted_pendulum': jp.arange(0, 6),
    'inverted_double_pendulum': jp.arange(0,5),
    'reacher': jp.arange(4,6),
    'reacherangle': jp.arange(4,6),
    'ur5e': jp.concatenate((jp.arange(0, 6), jp.arange(10, 34)), axis=0),
    'walker2d': jp.arange(0,11)

}

# Qvel and others
VELOCITY = {
    'acrobot': jp.arange(2, 4),
    'ant': jp.arange(13, 27),
    'fetch': jp.arange(49, 88),
    'grasp': jp.concatenate((jp.arange(56, 104), jp.arange(107, 110)), axis=0),
    'halfcheetah': jp.arange(11, 23,),
    'hopper': jp.arange(8, 14),
    'humanoid': jp.concatenate((jp.arange(22, 45), jp.arange(144, 210)), axis=0),
    'humanoidstandup': jp.concatenate((jp.arange(22, 45), jp.arange(144, 210)), axis=0),
    'inverted_pendulum': jp.arange(6, 10),
    'inverted_double_pendulum': jp.arange(5, 25),
    'reacher': jp.arange(6,8),
    'reacherangle': jp.arange(6,8),
    'ur5e': jp.arange(34, 58),
    'walker2d': jp.arange(11, 20)
}

# Position/local vector information about target
TARGET_POS = {
    'fetch': jp.arange(6, 10),
    'grasp': jp.arange(4, 8),
    'reacher': jp.concatenate((jp.arange(0, 4), jp.arange(8, 11)), axis=0),
    'reacherangle': jp.concatenate((jp.arange(0, 4), jp.arange(8, 11)), axis=0),
    'ur5e': jp.arange(6, 10)
}

# Position of object to be moved to target
OBJECT_POS = {
    'grasp': jp.arange(0, 4)
}

# Heading to target, obs, etc
HEADINGS = {
    'grasp': jp.concatenate((jp.arange(104, 107), jp.arange(110, 116)), axis=0),
}

#Contact forces (delta velocity, delta angle * bodies in system)
CFRC = {
    'ant': jp.arange(27, 87),
    'fetch': jp.arange(88, 101),
    'grasp': jp.arange(116, 132),
    'humanoid': jp.arange(210, 299),
    'humanoidstandup': jp.arange(210, 299),
    'ur5e': jp.arange(58, 66)
}