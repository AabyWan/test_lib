"""
The :mod:`phaser.similarities` module includes various ...
"""

from ._helpers import (
    find_inter_samplesize,
    IntraDistance,
    InterDistance
)

from ._distances import (
    cosine,
    hamming
)

__all__ = [
    "IntraDistance",
    "InterDistance",
    "find_inter_samplesize",
    "cosine",
    "hamming"
]