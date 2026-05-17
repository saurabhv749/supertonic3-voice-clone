from .default import TrainConfig
from .utterances import texts


# Explicitly define what is available when someone imports *
__all__ = [
    "TrainConfig",
    "texts"
]