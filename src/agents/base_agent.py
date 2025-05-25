"""
Base agent abstract class defining the common interface for all race strategy agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """Base class for all strategy agents in the F1 simulator."""

    def __init__(self, name: str):
        """
        Initialize the base agent.

        Args:
            name: The name of the agent
        """
        self.name = name
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and return outputs.

        Args:
            inputs: Dictionary of input data

        Returns:
            Dictionary of output data
        """
        pass

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        Update the agent's internal state.

        Args:
            new_state: Dictionary of state updates
        """
        self.state.update(new_state)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the agent's current state.

        Returns:
            The agent's state dictionary
        """
        return self.state

    def reset(self) -> None:
        """Reset the agent's state."""
        self.state = {}
