from math import sqrt


def cycle_interception(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> bool:
    cen_dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if not cen_dist:
        return r1 == r2
    else:
        return cen_dist <= r1 + r2
