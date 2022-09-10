#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np

###############################################################################

AverageableType = Union[int, float, np.ndarray]


def update_average(
    current_average: AverageableType,
    current_size: int,
    addition: AverageableType,
) -> AverageableType:
    return (current_size * current_average + addition) / (current_size + 1)


class IncrementalAverage:
    def __init__(self) -> None:
        self._current_mean: Optional[np.ndarray] = None
        self._current_size: int = 0

    @property
    def current_mean(self) -> Optional[np.ndarray]:
        return self._current_mean

    @property
    def current_size(self) -> int:
        return self._current_size

    def add(
        self,
        addition: np.ndarray,
    ) -> np.ndarray:
        # Store initial if nothing yet
        if self.current_mean is None:
            self._current_mean = addition
            self._current_size += 1
            return self.current_mean

        # Update if actual addition
        self._current_mean = update_average(
            self.current_mean,
            self.current_size,
            addition,
        )
        self._current_size += 1
        return self.current_mean
