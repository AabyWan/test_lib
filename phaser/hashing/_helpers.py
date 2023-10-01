import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm

## Local imports from ..utils
from ..utils import ImageLoader

class ComputeHashes:
    """Compute Perceptual Hashes using a defined dictionary of algorithms, \\
        and a corresponding list for transformations to be applies
    """

    def __init__(
        self,
        algorithms: dict,
        transformations: list,
        n_jobs=1,
        backend="loky",
        progress_bar=False,
    ) -> None:
        """_summary_

        Args:
            algorithms (dict): Dictionary containing {'phash': phaser.hashing.PHASH(<settings>)}
            transformations (list): A list of transformations to be applies [phaser.transformers.Flip(<setting>)]
            n_jobs (int, optional): How many CPU cores to use. -1 uses all resources. Defaults to 1.
            backend (str, optional): Pass backend parameter to joblib. Defaults to 'loky'.
        """
        self.algos = algorithms
        self.trans = transformations
        self.n_jobs = n_jobs
        self.backend = backend
        self.progress_bar = progress_bar

    def fit(self, paths: list) -> pd.DataFrame:
        """Run the computation

        Args:
            paths (list): A list of absolute paths to original images

        Returns:
            pd.DataFrame: Dataset containing all computations
        """
        hashes = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(sim_hashing)(
                img_path=p, algorithms=self.algos, transformations=self.trans
            )
            for p in tqdm(paths, desc="Files", disable=not self.progress_bar)
        )

        # joblib returns a list of numpy arrays from sim_hashing
        # the length depends on how many transformations are applied
        # concatenate the list and pass to a dataframe below
        hashes = np.concatenate(hashes)  # type:ignore

        # derive the column names based on the list of algorithms
        cols = ["filename", "transformation", *list(self.algos.keys())]
        df = pd.DataFrame(hashes, columns=cols)

        return df


def sim_hashing(img_path, transformations=[], algorithms={}):
    image_obj = ImageLoader(img_path)
    img = deepcopy(image_obj)

    outputs = []

    # loop over a set of algorithms
    hashes = [a.fit(img.image) for a in algorithms.values()]
    outputs.append([img.filename, "orig", *hashes])

    if len(transformations) > 0:
        for transform in transformations:
            _img = transform.fit(img)

            hashes = [a.fit(_img) for a in algorithms.values()]
            outputs.append([img.filename, transform.aug_name, *hashes])

    return np.row_stack(outputs)
