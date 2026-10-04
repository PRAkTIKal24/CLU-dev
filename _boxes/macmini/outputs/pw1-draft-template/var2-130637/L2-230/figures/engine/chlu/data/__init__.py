"""Data generators for CHLU experiments."""

from chlu.data.figure8 import generate_figure8
from chlu.data.sine_waves import generate_sine_waves, add_noise
from chlu.data.mnist import load_mnist_pca
from chlu.data.circle_vacuum import generate_circle_vacuum
from chlu.data.mqar import generate_mqar, make_token_embeddings

# ⭐ C3: the real-stream registry — a corpus is a CONFIG VALUE, not a code path.
# Adding a stream is one loader module + one register_corpus() call.
from chlu.data.corpora import (
    CorpusSpec,
    available_corpora,
    get_corpus,
    load_corpus,
    register_corpus,
    stage_corpus,
)

__all__ = [
    "CorpusSpec",
    "available_corpora",
    "get_corpus",
    "load_corpus",
    "register_corpus",
    "stage_corpus",
    "generate_figure8",
    "generate_sine_waves",
    "add_noise",
    "load_mnist_pca",
    "generate_circle_vacuum",
    "generate_mqar",
    "make_token_embeddings",
]
