"""

The :mod:`phaser.similarities` module includes various ...
"""


from ._helpers import (
    find_inter_samplesize,
    IntraDistance,
    InterDistance,
    validate_metrics,
)


from ._distances import test_synthetic


__all__ = ["IntraDistance", "InterDistance", "find_inter_samplesize", "test_sythetic"]
