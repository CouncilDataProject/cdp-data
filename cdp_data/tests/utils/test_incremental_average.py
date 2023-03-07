#!/usr/bin/env python

from typing import List, Union

import numpy as np
import pytest

from cdp_data.utils.incremental_average import (
    IncrementalAverage,
    IncrementalStats,
    update_average,
)


###############################################################################


@pytest.mark.parametrize(
    "additions, averages",
    [
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([2, 3, 4], [1.5, 2, 2.5]),
        ([2, 3, 4, 5, 4, 2], [1.5, 2, 2.5, 3, 19 / 6, 3]),
        ([0, 1, 0, 0, 0], [0.5, 2 / 3, 0.5, 0.4, 1 / 3]),
        (
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
        ),
        (
            [
                np.array([2, 2, 2]),
                np.array([3, 3, 3]),
            ],
            [
                np.array([1.5, 1.5, 1.5]),
                np.array([2, 2, 2]),
            ],
        ),
        (
            [
                np.array([2, 4, 6]),
                np.array([4, 6, 8]),
            ],
            [
                np.array([1.5, 2.5, 3.5]),
                np.array([7 / 3, 11 / 3, 5]),
            ],
        ),
    ],
)
def test_update_average(
    additions: List[Union[int, np.ndarray]],
    averages: List[Union[int, float, np.ndarray]],
) -> None:
    size = 1
    for i, addition in enumerate(additions):
        if i == 0:
            average = update_average(1, size, addition)
        else:
            average = update_average(average, size, addition)

        size += 1

        # Check that current average is correct
        np.testing.assert_equal(average, averages[i])


@pytest.mark.parametrize(
    "additions, averages",
    [
        (
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
        ),
        (
            [
                np.array([2, 2, 2]),
                np.array([3, 3, 3]),
            ],
            [
                np.array([1.5, 1.5, 1.5]),
                np.array([2, 2, 2]),
            ],
        ),
        (
            [
                np.array([2, 4, 6]),
                np.array([4, 6, 8]),
            ],
            [
                np.array([1.5, 2.5, 3.5]),
                np.array([7 / 3, 11 / 3, 5]),
            ],
        ),
    ],
)
def test_incremental_average(
    additions: List[np.ndarray],
    averages: List[np.ndarray],
) -> None:
    # Setup incremental averager
    inc_avg = IncrementalAverage()
    inc_avg.add(np.array([1, 1, 1]))

    # Iter additions
    for i, addition in enumerate(additions):
        average = inc_avg.add(addition)
        np.testing.assert_equal(average, averages[i])


@pytest.mark.parametrize(
    "additions, averages, mins, maxs",
    [
        (
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
        ),
        (
            [
                np.array([2, 2, 2]),
                np.array([3, 3, 3]),
            ],
            [
                np.array([1.5, 1.5, 1.5]),
                np.array([2, 2, 2]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([2, 2, 2]),
                np.array([3, 3, 3]),
            ],
        ),
        (
            [
                np.array([2, 4, 6]),
                np.array([4, 6, 8]),
            ],
            [
                np.array([1.5, 2.5, 3.5]),
                np.array([7 / 3, 11 / 3, 5]),
            ],
            [
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
            ],
            [
                np.array([2, 4, 6]),
                np.array([4, 6, 8]),
            ],
        ),
    ],
)
def test_incremental_stats(
    additions: List[np.ndarray],
    averages: List[np.ndarray],
    mins: List[np.ndarray],
    maxs: List[np.ndarray],
) -> None:
    # Setup incremental stats
    inc_stats = IncrementalStats()
    inc_stats.add(np.array([1, 1, 1]))

    # Iter additions
    for i, addition in enumerate(additions):
        inc_stats.add(addition)
        np.testing.assert_equal(inc_stats.current_mean, averages[i])
        np.testing.assert_equal(inc_stats.current_min, mins[i])
        np.testing.assert_equal(inc_stats.current_max, maxs[i])
