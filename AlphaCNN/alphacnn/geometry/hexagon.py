import numpy as np


def area2ri(area):
    return np.sqrt(area / (2. * np.sqrt(3.)))


def ri2area(ri):
    return 2. * np.sqrt(3.) * ri ** 2


def ri2ru(ri):
    return ri * 2. / np.sqrt(3.)


def ru2ri(ru):
    return ru * np.sqrt(3.) / 2.


def get_hex_coords(ri, center=(0., 0.)):
    ru = ri2ru(ri)
    hex_coords = np.array([
        [-ru / 2., +ri],
        [+ru / 2., +ri],
        [+ru, 0],
        [+ru / 2., -ri],
        [-ru / 2., -ri],
        [-ru, 0],
    ])

    hex_coords += np.array(center)

    return hex_coords


def get_square_coords(ri, center=(0., 0.)):
    square_coords = np.array([
        [-ri, -ri],
        [-ri, +ri],
        [+ri, +ri],
        [+ri, -ri],
    ])

    square_coords += np.array(center)

    return square_coords
