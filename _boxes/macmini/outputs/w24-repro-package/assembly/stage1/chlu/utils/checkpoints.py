"""Model checkpointing utilities for CHLU."""
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import equinox as eqx
def save_model(
    model, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model to disk using Equinox's filter + pickle approach.

    Args:
        model: Model to save (CHLU, NeuralODE, or LSTMPredictor)
        path: Path to save the model (.pkl extension recommended)
        metadata: Optional metadata dictionary (config, metrics, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Separate trainable parameters from static structure
    params, static = eqx.partition(model, eqx.is_array)
    
    checkpoint = {
        "params": params, 
        "static": static,
        "metadata": metadata or {}
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_model(path: Union[str, Path], model_template: Optional[Any] = None) -> Any:
    """Load a model: a pickle written by this package, or a shipped array-only file (.eqx)."""
    path = Path(path)
    if not path.exists() and path.with_suffix(".eqx").exists():
        from ..weights import load_eqx
        return load_eqx(path.with_suffix(".eqx"))[0]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    return eqx.combine(checkpoint["params"], checkpoint["static"])


