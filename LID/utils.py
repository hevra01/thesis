from typing import Iterable, Tuple

import numpy as np


def convex_hull(
    x: Iterable,
    y: Iterable,
) -> Tuple[np.array, np.array]:
    """
    A linear algorithm to find the convex hull of a given curve, it
    returns two numpy arrays denoting the x and y points of the convex hull.
    """

    # a stack will contain the x and y values,
    # when a new (x, y) is encountered, it is compared
    # to the last and second to last element of the
    # convex hull and pops the last element if that point
    # lies above the intercept of the second to last point
    # and the new point
    # Start with the first point on the curve as the beginning of the hull.
    stack_x = [x[0]] 
    stack_y = [y[0]]
    for i in range(1, len(x)):
        if y[i] >= stack_y[-1]:
            stack_x.append(x[i])
            stack_y.append(y[i])
            continue

        while len(stack_x) > 1:
            x_ref, y_ref = stack_x[-2], stack_y[-2]
            x_bef, y_bef = stack_x[-1], stack_y[-1]
            x_cur, y_cur = x[i], y[i]

            theta = (x_bef - x_ref) / (x_cur - x_ref)

            y_compare = y_ref + theta * (y_cur - y_ref)
            if y_bef > y_compare:
                stack_x.pop()
                stack_y.pop()
            else:
                break

        stack_x.append(x[i])
        stack_y.append(y[i])

    # after that, the x values that were popped should return back to the
    # list with linear interpolation.
    final_x = [x[0]]
    final_y = [y[0]]
    pnt = 1
    for i in range(1, len(stack_x)):
        while x[pnt] < stack_x[i]:
            lst_x = final_x[-1]
            lst_y = final_y[-1]
            nxt_x = stack_x[i]
            nxt_y = stack_y[i]

            final_x.append(x[pnt])
            final_y.append((x[pnt] - lst_x) / (nxt_x - lst_x) * (nxt_y - lst_y) + lst_y)
            pnt += 1
        final_x.append(stack_x[i])
        final_y.append(stack_y[i])
        pnt += 1

    return np.array(final_x), np.array(final_y)