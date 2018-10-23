import numpy as np

DEFAULT_SIZE_X = 0.0
DEFAULT_SIZE_Y = 0.0

index_alpha_x = 0.0075
thumb_alpha_x = 0.0073

index_sigma_a_x = 1.241
thumb_sigma_a_x = 1.349

index_alpha_y = 0.0104
thumb_alpha_y = 0.0113

index_sigma_a_y = 1.118
thumb_sigma_a_y = 1.179


def index_sigma_x(size=DEFAULT_SIZE_X):
    return np.sqrt(index_alpha_x * size**2 + index_sigma_a_x**2)


def thumb_sigma_x(size=DEFAULT_SIZE_X):
    return np.sqrt(thumb_alpha_x * size**2 + thumb_sigma_a_x**2)


def index_sigma_y(size=DEFAULT_SIZE_Y):
    return np.sqrt(index_alpha_y * size**2 + index_sigma_a_y**2)


def thumb_sigma_y(size=DEFAULT_SIZE_Y):
    return np.sqrt(thumb_alpha_y * size**2 + thumb_sigma_a_y**2)


def cartesian_index_probdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = index_sigma_x(size_x)
    s_y = index_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)


def cartesian_thumb_probdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = thumb_sigma_x(size_x)
    s_y = thumb_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)
