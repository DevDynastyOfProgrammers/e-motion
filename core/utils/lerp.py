def lerp(start: float, end: float, factor: float) -> float:
    """
    Performs linear interpolation between two values.

    :param start: The starting value.
    :param end: The target value.
    :param factor: The interpolation factor, typically between 0.0 and 1.0.
                   A value of 0.0 returns start, 1.0 returns end.
    :return: The interpolated value.
    """
    return start + (end - start) * factor