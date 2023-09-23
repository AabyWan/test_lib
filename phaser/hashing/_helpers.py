import logging, pathlib, os
import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm.auto import tqdm

## Local imports from ..utils
from ..utils import ImageLoader


pathlib.Path("./logs").mkdir(exist_ok=True)
logging.basicConfig(filename='./logs/process.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
        dirpath = os.path.dirname(paths[0])
        logging.info(f"===Beginning to process directory {dirpath}===")

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

        # remove rows with any nan values
        pre_clean_count = len(df)
        df.dropna(how='any', inplace=True, axis=0)  
        post_clean_count = len(df)

        num_removed = pre_clean_count - post_clean_count
        num_versions = 1 + len(self.trans) # original + transofmrations
        if num_removed :
            logging.info(f"Dropped null records for {int(num_removed/num_versions)} files. ")

        return df


def sim_hashing(img_path, transformations=[], algorithms={}):

    error = False

    try:
        image_obj = ImageLoader(img_path)
        img = deepcopy(image_obj)
    except Exception as err:
        logging.error(f"Error processing path {img_path}: {err}")
        error = True

    outputs = []
    if not error:
        # loop over a set of algorithms
        hashes = [a.fit(img.image) for a in algorithms.values()]
        outputs.append([img.filename, "orig", *hashes])

        if len(transformations) > 0:
            for transform in transformations:
                _img = transform.fit(img)

                hashes = [a.fit(_img) for a in algorithms.values()]
                outputs.append([img.filename, transform.aug_name, *hashes])

    else:
        # An error occured, so we want to abandon this set of observations.
        # Generate stacked np.nan arrays as placeholders to remove later.
        hashes = [None] * len(algorithms) # each hash is replaced by NAN
        outputs.append([img_path, "orig", *hashes])
        for transform in transformations:
            outputs.append([img_path, transform.aug_name, *hashes]) # one set of NAN hashes for each transformation

    return np.row_stack(outputs)
