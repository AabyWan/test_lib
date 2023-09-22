"""
Module to define custom distances not already present in scipt.spatial.distance.
Distance metrics should return a normalised value between 0 and 1, and take two vectors (u,v by convention in scipy), and possibly other kwargs, as input.
"""

# Imports for test synthtic distance generator
import sys
import numpy as np


def test_synthetic(u=None, v=None):
    """
    A dummy distance metric to simulate "good" performance.
    Performs caller introspection to determine which distribution to draw from.
    """

    # /Slightly/ hacky caller introspection
    caller_name = sys._getframe().f_back.f_code.co_name

    # In this library:
    # intra-distance are measured using scipy.spatial.distance.cdist
    # inter-distance are measured using scipy.spatial.distance.pdist
    # Check if we are doing intra-distances and draw from a pareto distribution
    # Otherwise, draw from a normal distribution

    if "cdist" in caller_name.lower():
        # Assume Intra-distance, pareto distribution largely falling near 0
        # Simulate good matching with some variance
        a = 10
        m = 0.1
        return np.random.pareto(a) * m

    else:
        # Simulate a tight normal distribution around 0.5
        mu = 0.5
        sigma = 0.05
        rng = np.random.default_rng()
        return rng.normal(mu, sigma)


# Keep track of distance metrics, add here and import in __init__.py when creating new distance metrics.
__DISTANCE_METRICS__ = [test_synthetic.__name__]
