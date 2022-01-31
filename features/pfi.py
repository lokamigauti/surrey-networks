import numpy as np

def calc_pfi(wd, sd, wd_unit='degrees', sd_unit='degrees'):
    """
    Return an np.array with [1, n] dimensions
    containing the difference between the wind direction and the direction of the n known sources

    :param wd: np.array of Wind direction in degrees or lat/lon vector
    :param sd: np.array of source direction in degrees
    :param wd_unit: string of wd unit. Can be 'degrees', 'radians', or 'vector'
    :param sd_unit: analogous as wd_unit for sd, but do not accept vector
    :return: direction difference in radians
    """
    # Format wd
    if wd_unit == 'degrees':
        wd = wd * np.pi / 180
    if wd_unit == 'vector':
        wd = np.arctan2(wd[1], [0])
    wd = wd % (2 * np.pi)  # reduces angle between 0 and 2 pi

    # Format sd
    if sd_unit == 'degrees':
        sd = sd * np.pi / 180
    sd = sd % (2 * np.pi)  # reduces angle between 0 and 2 pi

    return sd - wd
