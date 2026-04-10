from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CameraInterface(ABC):
    """Minimal camera interface expected by the robot-side control loop."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Return one RGB frame with shape (H, W, 3) and dtype uint8."""


class RobotInterface(ABC):
    """Minimal robot SDK interface expected by the robot-side control loop."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the robot controller."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot controller."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Return the latest robot-side observation."""

    @abstractmethod
    def send_action(self, action: dict[str, Any]) -> None:
        """Execute one low-level action dictionary on the robot."""

    def reset(self) -> None:
        """Optional reset hook."""
