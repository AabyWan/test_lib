import numpy as np
import pandas as pd
import phaser.similarities
from scipy.spatial import distance as dist
from scipy.spatial.distance import cdist, pdist
from itertools import combinations
from tqdm import tqdm
from typing import Callable

def find_inter_samplesize(num_images: int) -> int:
    for n in range(0, num_images):
        if (n * (n - 1)) / 2 > num_images:
            return n
    return 0

def validate_metrics(metrics: dict) -> bool:
    """Function to perform light validation of metrics.
    If the distance provided is a string, it should be in scipy.spatial.distance.
    If a custom metric is provided, we at least check to see that it is callable.
    If either of the above conditions are not met, or the metrics list is empty, throw an Exception.
    """
    if not metrics:
        raise (Exception("No distance metrics specified."))

    invalid = []

    for mname, value in metrics.items():
        if isinstance(value, str):
            if value not in dist._METRICS_NAMES: #type:ignore
                invalid.append(
                    f"{mname} does not match the name of a distance metric in scipy.spatial.distance."
                )
        elif not isinstance(value, Callable):
            invalid.append(f"{mname} is not a valid Callable object.")
        elif isinstance(value, Callable):
            if (
                value.__name__
                not in phaser.similarities._distances.__DISTANCE_METRICS__
            ):
                invalid.append(
                    f"{mname} does not appear to be a valid distance function in phaser.similarities"
                )

    # If no invalid items, all entries in the list pass basic check.
    if not invalid:
        return True
    else:
        # There are invalid objects, avoid running the tests.
        message = f"Invalid metrics found:{invalid}"
        raise (Exception(message))

# DISTANCE COMPUTATION
class IntraDistance:
    def __init__(
        self,
        le, 
        dist_w=None, # would expect a dicitonary
        distance_metrics={}, 
        set_class=0, 
        progress_bar=False):
        #
        self.le = le 
        self.dist_w = dist_w
        self.distance_metrics = distance_metrics
        self.set_class = set_class
        self.progress_bar = progress_bar

        validate_metrics(self.distance_metrics)

    def intradistance(self, x, algorithm, metric, weights):
        # store the first hash and reshape into 2d array as required by cdist func.
        xa = x[algorithm].iloc[0].reshape(1, -1)

        # row stack the other hashes
        xb = x.iloc[1:][algorithm].values
        xb = np.row_stack(xb)

        # Get the vlaue corresponding to the metric key.
        # This is either a string representing a name from scipy.spatial.distances
        # or a callable function implementing another metric.
        metric_value = self.distance_metrics[metric]
        
        return cdist(xa, xb, metric=metric_value, w=weights)

    def fit(self, data):
        self.files_ = data["filename"].unique()
        self.n_files_ = len(self.files_)

        distances = []

        for a in tqdm(self.le['a'].classes_, disable=not self.progress_bar, desc="Hash"):
            for m in self.le['m'].classes_:
                
                if self.dist_w:
                    w = self.dist_w[f"{a}_{m}"]
                else: w=None
                
                # Compute the distances for each filename
                grp_dists = data.groupby(["filename"]).apply(
                    self.intradistance, 
                    algorithm=a, 
                    metric=m,
                    weights=w
                )

                # Stack each distance into rows
                grp_dists = np.row_stack(grp_dists)

                # Get the integer labels for algo and metric
                a_label = self.le['a'].transform(a.ravel())[0]
                m_label = self.le['m'].transform(m.ravel())

                grp_dists = np.column_stack(
                    [
                        self.files_,  # fileA
                        self.files_,  # fileB (same in intra!)
                        np.repeat(a_label, self.n_files_),
                        np.repeat(m_label, self.n_files_),
                        np.repeat(self.set_class, self.n_files_),
                        grp_dists,
                    ]
                )
                distances.append(grp_dists)

        distances = np.concatenate(distances)

        # Create the dataframe output
        cols = ["fileA", "fileB", "algo", "metric", "class", *self.le['t'].classes_[:-1]]
        distances = pd.DataFrame(distances, columns=cols)
        distances["orig"] = 0

        # set int columns accordingly
        int_cols = cols[:5]
        distances[int_cols] = distances[int_cols].astype(int)

        # Convert distances to similarities
        sim_cols = distances.columns[5:]
        distances[sim_cols] = 1 - distances[sim_cols]

        return distances


class InterDistance:
    def __init__(
        self,
        le,
        dist_w=None,
        distance_metrics={},
        set_class=1,
        n_samples=100,
        random_state=42,
        progress_bar=False,
    ):
        self.le = le
        self.dist_w = dist_w
        self.distance_metrics = distance_metrics
        self.set_class = set_class
        self.n_samples = n_samples
        self.random_state = random_state
        self.progress_bar = progress_bar

        validate_metrics(self.distance_metrics)

    def interdistance(self, x, algorithm, metric, weights):
        # get hashes into a 2d array
        hashes = np.row_stack(x[algorithm])

        # Get the vlaue corresponding to the metric key.
        # This is either a string representing a name from scipy.spatial.distances
        # or a callable function implementing another metric.
        metric_value = self.distance_metrics[metric]

        # return pairwise distances of all combinations
        return pdist(hashes, metric_value, w=weights)

    def fit(self, data):
        # Get the label used to encode 'orig'
        orig_label = self.le['t'].transform(np.array(["orig"]).ravel())[0]

        # Assert sufficient data to sample from.
        assert len(data[data["transformation"] == orig_label]) >= self.n_samples

        # Pick the samples
        self.samples_ = (
            data[data["transformation"] == orig_label]
            .sample(self.n_samples, random_state=self.random_state)["filename"]
            .values
        )

        # Subset the data
        subset = data[data["filename"].isin(self.samples_)]

        # Create unique pairs matching the output of scipy.spatial.distances.pdist
        self.pairs_ = np.array(
            [c for c in combinations(subset["filename"].unique(), 2)]
        )

        # Count the number of unique pairs
        self.n_pairs_ = len(self.pairs_)

        # List to hold distances while looping over algorithms and metrics
        distances = []

        # Do the math using Pandas groupby
        for a in tqdm(self.le['a'].classes_, disable=not self.progress_bar, desc="Hash"):
            for m in self.le['m'].classes_:

                if self.dist_w:
                    w = self.dist_w[f"{a}_{m}"]
                else: w=None
                
                # Compute distances for each group of transformations
                grp_dists = subset.groupby(["transformation"]).apply(
                    self.interdistance,  # type:ignore
                    algorithm=a,
                    metric=m,
                    weights=w
                )

                # Transpose to create rows of observations
                X_dists = np.transpose(np.row_stack(grp_dists.values))

                # Get the integer labels for algo and metric
                a_label = self.le['a'].transform(a.ravel())[0]
                m_label = self.le['m'].transform(m.ravel())

                # Add columns with pairs of the compared observations
                X_dists = np.column_stack(
                    [
                        self.pairs_,
                        np.repeat(a_label, self.n_pairs_),
                        np.repeat(m_label, self.n_pairs_),
                        np.repeat(self.set_class, self.n_pairs_),
                        X_dists,
                    ]
                )

                # Add the results to the distances array
                distances.append(X_dists)

        # Flatten the distances array
        distances = np.concatenate(distances)

        # Create the dataframe output
        cols = ["fileA", "fileB", "algo", "metric", "class", *self.le['t'].classes_]
        distances = pd.DataFrame(distances, columns=cols)

        # Set datatype to int on all non-distance columns
        int_cols = cols[:5]
        distances[int_cols] = distances[int_cols].astype(int)

        # Convert distances to similarities
        sim_cols = distances.columns[5:]
        distances[sim_cols] = 1 - distances[sim_cols]

        return distances
