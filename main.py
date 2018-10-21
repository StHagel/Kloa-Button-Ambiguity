import numpy as np
from scipy.integrate import dblquad

DEFAULT_SIZE_X = 0.0
DEFAULT_SIZE_Y = 0.0

# The default size of the buttons in x and y direction respectively, given in mm.

generic_alpha_x = 0.0075
index_alpha_x = 0.0075
thumb_alpha_x = 0.0073

generic_sigma_a_x = 1.296
index_sigma_a_x = 1.241
thumb_sigma_a_x = 1.349

generic_alpha_y = 0.0108
index_alpha_y = 0.0104
thumb_alpha_y = 0.0113

generic_sigma_a_y = 1.153
index_sigma_a_y = 1.118
thumb_sigma_a_y = 1.179

alpha_x = [generic_alpha_x, index_alpha_x, thumb_alpha_x]
sigma_a_x = [generic_sigma_a_x, index_sigma_a_x, thumb_sigma_a_x]
alpha_y = [generic_alpha_y, index_alpha_y, thumb_alpha_y]
sigma_a_y = [generic_sigma_a_y, index_sigma_a_y, thumb_sigma_a_y]
# These are the values taken from the Google paper.


def generic_sigma_x(size=DEFAULT_SIZE_X):
    return np.sqrt(generic_alpha_x * size**2 + generic_sigma_a_x**2)


def index_sigma_x(size=DEFAULT_SIZE_X):
    return np.sqrt(index_alpha_x * size**2 + index_sigma_a_x**2)


def thumb_sigma_x(size=DEFAULT_SIZE_X):
    return np.sqrt(thumb_alpha_x * size**2 + thumb_sigma_a_x**2)


def generic_sigma_y(size=DEFAULT_SIZE_Y):
    return np.sqrt(generic_alpha_y * size**2 + generic_sigma_a_y**2)


def index_sigma_y(size=DEFAULT_SIZE_Y):
    return np.sqrt(index_alpha_y * size**2 + index_sigma_a_y**2)


def thumb_sigma_y(size=DEFAULT_SIZE_Y):
    return np.sqrt(thumb_alpha_y * size**2 + thumb_sigma_a_y**2)


def generic_propdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = generic_sigma_x(size_x)
    s_y = generic_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)


def index_propdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = index_sigma_x(size_x)
    s_y = index_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)


def thumb_propdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = thumb_sigma_x(size_x)
    s_y = thumb_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)

# These functions return the probability distruibution at the point (x,y) on a button of a given size under the
# assumption that the sigmas in x an dy direction are not correlated and with mu_x and mu_y set to zero.


# As a first test of the model we will calculate the generic probability to hit a button on Kloa and qwerty
# respectively. The size of Kloas has been measured by hand from the printout, the qwerty keyboard used for reference
# is the UK layout of the standard keyboard in Android 8.1.0 with Lineage OS 15.1 on a Google Nexus 5X phone.

size_x_kloa = 11.0
size_y_kloa = 11.0
generic_prob_kloa = dblquad(generic_propdist, -size_y_kloa / 2.0, size_y_kloa / 2.0, lambda x: -size_x_kloa / 2.0,
                            lambda x: size_x_kloa / 2.0, args=(size_x_kloa, size_y_kloa))[0]

size_x_qwerty = 7.0
size_y_qwerty = 8.0
generic_prob_qwerty = dblquad(generic_propdist, -size_y_qwerty / 2.0, size_y_qwerty / 2.0,
                              lambda x: -size_x_qwerty / 2.0, lambda x: size_x_qwerty / 2.0,
                              args=(size_x_qwerty, size_y_qwerty))[0]

print("Chance to miss the button:")
print("Kloa: " + str(100.0 * (1.0 - generic_prob_kloa)) + "%")
print("Qwerty: " + str(100.0 * (1.0 - generic_prob_qwerty)) + "%\n")

print("Ratio Qwerty/Kloa:")
print(str((1.0 - generic_prob_qwerty) / (1.0 - generic_prob_kloa)))




