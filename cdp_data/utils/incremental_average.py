#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np

###############################################################################

AverageableType = Union[int, float, np.ndarray]

###############################################################################


def update_average(
    current_average: AverageableType,
    current_size: int,
    addition: AverageableType,
) -> AverageableType:
    return (current_size * current_average + addition) / (current_size + 1)


class IncrementalAverage:
    def __init__(self) -> None:
        self._current_mean: AverageableType = None
        self._current_size: int = 0

    @property
    def current_mean(self) -> Optional[AverageableType]:
        return self._current_mean

    @property
    def current_size(self) -> int:
        return self._current_size

    def add(
        self,
        addition: AverageableType,
    ) -> AverageableType:
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

    def __str__(self) -> str:
        return f"<IncrementalAverage [current_size: {self.current_size}]>"

    def __repr__(self) -> str:
        return str(self)


class IncrementalStats:
    def __init__(self) -> None:
        self._current_mean: Optional[AverageableType] = None
        self._current_size: int = 0
        self._current_max: Optional[AverageableType] = None
        self._current_min: Optional[AverageableType] = None

    @property
    def current_mean(self) -> Optional[AverageableType]:
        return self._current_mean

    @property
    def current_size(self) -> int:
        return self._current_size

    @property
    def current_max(self) -> Optional[AverageableType]:
        return self._current_max

    @property
    def current_min(self) -> Optional[AverageableType]:
        return self._current_min

    def add(
        self,
        addition: AverageableType,
    ) -> "IncrementalStats":
        # Store initial mean if not set or update
        if self.current_mean is None:
            self._current_mean = addition
        else:
            self._current_mean = update_average(
                self.current_mean,
                self.current_size,
                addition,
            )

        # Update size regardless
        self._current_size += 1

        # Store initial max if not set or update
        if self.current_max is None:
            self._current_max = addition
        else:
            self._current_max = np.maximum(self.current_max, addition)

        # Store initial min if not set or update
        if self.current_min is None:
            self._current_min = addition
        else:
            self._current_min = np.minimum(self.current_min, addition)

        return self

    def __str__(self) -> str:
        return f"<IncrementalStats [current_size: {self.current_size}]>"

    def __repr__(self) -> str:
        return str(self)
