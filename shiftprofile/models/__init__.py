"""Model registration.

Importing this package registers every vision model under a stable name,
so configs and result records refer to models as strings.
"""

from ..registry import MODELS
from .vision import build_model

# Register the vision models
# Note: resnet18 and resnet18_augmix are the SAME architecture.
# The difference is training-time augmentation only (AugMix data augmentation),
# which lives in the data module. This is made explicit here because a reader
# might otherwise look for an architectural difference that does not exist.
MODELS.register("resnet18")(lambda **kwargs: build_model("resnet18", **kwargs))
MODELS.register("resnet18_augmix")(lambda **kwargs: build_model("resnet18_augmix", **kwargs))
MODELS.register("vit_tiny")(lambda **kwargs: build_model("vit_tiny", **kwargs))

__all__ = ["build_model"]
