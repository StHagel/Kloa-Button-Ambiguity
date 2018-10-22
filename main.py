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


def cartesian_generic_probdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = generic_sigma_x(size_x)
    s_y = generic_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)


def cartesian_index_probdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = index_sigma_x(size_x)
    s_y = index_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)


def cartesian_thumb_probdist(x, y, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    s_x = thumb_sigma_x(size_x)
    s_y = thumb_sigma_y(size_y)
    return np.exp(-0.5 * (x / s_x) ** 2 - 0.5 * (y / s_y) ** 2) / (2.0 * np.pi * s_x * s_y)


def polar_generic_probdist(r, phi, size_x=DEFAULT_SIZE_X, size_y=DEFAULT_SIZE_Y):
    return r * cartesian_generic_probdist(r * np.cos(phi), r * np.sin(phi), size_x, size_y)


def int_probdist(r_low, r_up):
    return dblquad(polar_generic_probdist, 0.0, 2.0 * np.pi,
                   lambda r: r_low, lambda r: r_up, args=(size_x_qwerty, size_y_qwerty))[0]
# These functions return the probability distruibution at the point (x,y) on a button of a given size under the
# assumption that the sigmas in x an dy direction are not correlated and with mu_x and mu_y set to zero.


# As a first test of the model we will calculate the generic probability to hit a button on Kloa and qwerty
# respectively. The size of Kloas has been measured by hand from the printout, the qwerty keyboard used for reference
# is the UK layout of the standard keyboard in Android 8.1.0 with Lineage OS 15.1 on a Google Nexus 5X phone.

# size_x_kloa = 11.0
# size_y_kloa = 11.0
# generic_prob_kloa = dblquad(cartesian_generic_probdist, -size_y_kloa / 2.0, size_y_kloa / 2.0,
#                             lambda x: -size_x_kloa / 2.0,
#                             lambda x: size_x_kloa / 2.0, args=(size_x_kloa, size_y_kloa))[0]

size_x_qwerty = 7.0
size_y_qwerty = 8.0

r1 = size_x_qwerty / 2.0
r2 = size_y_qwerty / 2.0
r3 = np.sqrt(size_x_qwerty ** 2 + size_y_qwerty ** 2) / 2.0
r4 = np.sqrt(size_x_qwerty ** 2 + (size_y_qwerty / 2.0) ** 2)
r5 = 1.5 * size_x_qwerty
r6 = 1.5 * size_y_qwerty
# These are the distances from the center of the g-Button on a 26 letter keyboard to:
#  - f and h (r1)
#  - v, t and y (r2)
#  - c and b (r3)
#  - r and u (r4)
#  - d and j (r5)
#  - the upper end of the keyboard / the spacebar (r6)

ambi = 1.0 * int_probdist(0, r1) + 3.0 * int_probdist(r1, r2) + 6.0 * int_probdist(r2, r3) \
    + 8.0 * int_probdist(r3, r4) + 10.0 * int_probdist(r4, r5) + 12.0 * int_probdist(r5, r6)
# We need to sum up all ambiguities and their respective probabilities

print(str(ambi))


