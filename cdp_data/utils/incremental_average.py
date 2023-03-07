#!/usr/bin/env python

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
        """Get the current mean."""
        return self._current_mean

    @property
    def current_size(self) -> int:
        """Get the number of objects that have been averaged."""
        return self._current_size

    def add(
        self,
        addition: AverageableType,
    ) -> AverageableType:
        """Add a new value to the average."""
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
        """Print details."""
        return f"<IncrementalAverage [current_size: {self.current_size}]>"

    def __repr__(self) -> str:
        """Get the object representation."""
        return str(self)


class IncrementalStats:
    def __init__(self) -> None:
        self._current_mean: Optional[AverageableType] = None
        self._current_size: int = 0
        self._current_max: Optional[AverageableType] = None
        self._current_min: Optional[AverageableType] = None

    @property
    def current_mean(self) -> Optional[AverageableType]:
        """Get the current mean."""
        return self._current_mean

    @property
    def current_size(self) -> int:
        """Get the number of objects that have been managed."""
        return self._current_size

    @property
    def current_max(self) -> Optional[AverageableType]:
        """Get the current max."""
        return self._current_max

    @property
    def current_min(self) -> Optional[AverageableType]:
        """Get the current min."""
        return self._current_min

    def add(
        self,
        addition: AverageableType,
    ) -> "IncrementalStats":
        """Add a new value to mean (and check for new max and min)."""
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
        """Print details."""
        return f"<IncrementalStats [current_size: {self.current_size}]>"

    def __repr__(self) -> str:
        """Get the object representation."""
        return str(self)
