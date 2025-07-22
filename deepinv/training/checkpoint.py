from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Checkpoint:
    """A wrapper around checkpoint dictionaries."""
    data: dict

    @classmethod
    def from_dict(data: dict) -> Checkpoint:
        """Wrap a dictionary into a Checkpoint instance.

        :raises AssertionError: If the input is not a dictionary or does not contain 'state_dict'.
        """
        assert isinstance(data, dict), f"Expected a dictionary, got {type(data).__name__}"
        assert "state_dict" in data, "Checkpoint data must contain 'state_dict' key"
        return Checkpoint(d)

    def to_dict(self) -> dict:
        """Returns a reference to the original data dictionary."""
        return self.data

    @property
    def state_dict(self) -> Any:
        """Returns the state dictionary from the checkpoint data.

        :raises KeyError: If 'state_dict' is not present in the checkpoint data.
        """
        return self.data["state_dict"]

    @property
    def optimizer(self) -> Any | None:
        """Returns the optimizer state if available, otherwise None."""
        return self.data.get("optimizer", None)

    @property
    def scheduler(self) -> Any | None:
        """Returns the scheduler state if available, otherwise None."""
        return self.data.get("scheduler", None)

    @property
    def wandb_id(self) -> Any | None:
        """Return the Weights & Biases ID of the run associated with the checkpoint if available, otherwise None."""
        return self.data.get("wandb_id", None)

    @property
    def epoch(self) -> Any | None:
        """Returns the epoch at which the checkpoint was generated, if available."""
        return self.data.get("epoch", None)


def load(path: Any, map_location: Any = "cpu") -> Checkpoint:
    """Load a checkpoint from the specified path."""
    data = torch.load(path, map_location=map_location, weights_only=False)
    return Checkpoint.from_dict(data)
