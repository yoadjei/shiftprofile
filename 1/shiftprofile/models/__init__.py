"""Model registration.

Importing this package registers every vision and tabular model under a
stable name, so configs and result records refer to models as strings.
"""

from ..registry import MODELS
from .vision import build_model
from .tabular import build_tabular_model, fit_tabular, predict_proba_tabular

# Register the vision models
# Note: resnet18 and resnet18_augmix are the SAME architecture.
# The difference is training-time augmentation only (AugMix data augmentation),
# which lives in the data module. This is made explicit here because a reader
# might otherwise look for an architectural difference that does not exist.
MODELS.register("resnet18")(lambda **kwargs: build_model("resnet18", **kwargs))
MODELS.register("resnet18_augmix")(lambda **kwargs: build_model("resnet18_augmix", **kwargs))
MODELS.register("vit_tiny")(lambda **kwargs: build_model("vit_tiny", **kwargs))

# Register the tabular models
MODELS.register("logreg")(lambda **kwargs: build_tabular_model("logreg", **kwargs))
MODELS.register("xgboost")(lambda **kwargs: build_tabular_model("xgboost", **kwargs))
MODELS.register("mlp")(lambda **kwargs: build_tabular_model("mlp", **kwargs))

__all__ = ["build_model", "build_tabular_model", "fit_tabular", "predict_proba_tabular"]
