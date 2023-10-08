import numpy as np
import pandas as pd

from sklearn.utils import compute_sample_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# For EER calculations
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from joblib import Parallel, delayed
from tqdm import tqdm

def calc_eer(fpr: np.ndarray, tpr: np.ndarray, threshold: np.ndarray):
    """
    Discovers the threshold where FPR and FRR intersects.

    Parameters
    ----------
    fpr : np.ndarray
        Array with False Positive Rate from sklearn.metrics.roc_curve
    tpr : np.ndarray
        Array with True Positive Rate from sklearn.metrics.roc_curve
    threshold : np.ndarray
        Array with thresholds from sklearn.metrics.roc_curve

    Returns
    -------
    floats, float
        eer_score, eer_threshold


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> eer_score, eer_threshold = calc_eer(fpr, tpr, thresholds)


    """
    # Implementation from -> https://yangcha.github.io/EER-ROC/
    # first position is always set to max_threshold+1 (close to 2) by sklearn,
    # overwrite with 1.0 to avoid EER threshold exceeding 1.0.
    # threshold[0] = 1.0
    eer_score = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_thresh = interp1d(fpr, threshold)(eer_score)

    return eer_score, float(eer_thresh)


def pred_at_threshold(y_scores, threshold, pos_label=1):
    # Make predictions based on a specific decision threshold
    # y_scores : array with predicted probabilities, similarities, or distances.
    # threshold : the specified threshold to seperate the two classes.
    # pos_label : integer defining the positive class.
    if pos_label == 0:
        return np.array((y_scores <= threshold)).astype(int)
    else:
        assert pos_label == 1
        return np.array((y_scores >= threshold)).astype(int)

class MetricMaker:
    def __init__(self, y_true, y_similarity, weighted=True) -> None:
        self.y_true = y_true
        self.y_sims = y_similarity
        self.weighted = weighted

        # call the fit function when instantiated
        self._fit()

    def _fit(self):
        # Create balanced sample weights for imbalanced evaluation
        if self.weighted:
            self.smpl_w = compute_sample_weight(class_weight="balanced", y=self.y_true)
        else:
            self.smpl_w = None

        # Compute the FPR, TPR, and Thresholds
        self.fpr, self.tpr, self.thresholds = roc_curve(
            y_true=self.y_true, y_score=self.y_sims, sample_weight=self.smpl_w
        )

        # Compute the AUC score
        self.auc = auc(self.fpr, self.tpr)
        self.eer_score, self.eer_thresh = calc_eer(self.fpr, self.tpr, self.thresholds)

    def get_fpr_threshold(self, max_fpr):
        return np.interp(max_fpr, self.fpr, self.thresholds)

    def get_cm(self, threshold, normalize="true", breakdown=False):
        """
        Compute and returns the Confusion Matrix at a certain decision threshold

        breakdown: instead returns tn, fp, fn, tp
        """
        # an ugly patch to allow passing 'none' to sklean arg
        if normalize == "none":
            normalize = None

        self.y_pred = pred_at_threshold(self.y_sims, threshold, pos_label=1)

        cm = confusion_matrix(
            y_true=self.y_true,
            y_pred=self.y_pred,
            sample_weight=self.smpl_w,
            normalize=normalize,  # type:ignore
        )

        if breakdown:
            tn, fp, fn, tp = cm.ravel()
            return tn, fp, fn, tp
        else:
            return cm

def makepretty(styler, **kwargs):
    # https://pandas.pydata.org/docs/user_guide/style.html#Styler-Object-and-Customising-the-Display
    title = kwargs["title"]
    styler.set_caption(f"Stats for '{title}'")

    styler.format(precision=4, thousands=".", decimal=",")
    styler.background_gradient(
        axis=None, subset=["25%", "75%"], vmin=0, vmax=1, cmap="Greys"
    )
    styler.hide(subset=["count"], axis=1)
    styler.format_index(str.upper, axis=1)

    return styler

def dist_stats(data, le, transform, style=True):
    stats = data.groupby(["algo", "metric"])[transform].describe().reset_index()
    stats["algo"] = le['a'].inverse_transform(stats["algo"])
    stats["metric"] = le['m'].inverse_transform(stats["metric"])

    if style:
        stats = stats.style.pipe(makepretty, title=transform)

    return stats


class BitAnalyzer:
    def __init__(self, df_h:pd.DataFrame, le_t:LabelEncoder) -> None:
        """
        Interface for analysing bit changes on triplet subsets.

        Parameters
        ----------
        df_h : pd.DataFrame
            Dataframe containing the raw hashes from the experiment
        le_t : LabelEncoder
            Label encoder for transformations

        Example use:
        ------------
        Assuming a subset dataframe with a defined distances 
        >>> from phaser.evaluation import MetricMaker, BitAnalyzer 
        >>> y_true = subset['class']
        >>> y_sims = subset[t_s]
        >>> mm = MetricMaker(y_true, y_sims, weighted=False)
        >>> cm = mm.get_cm(mm.eer_thresh, normalize='none')
        >>> BA = BitAnalyzer(df_hashes, le_t)
        >>> BA.fit(subset, mm.y_pred, t_l, a_s)
        """
        # Dataframe with all the orignal hashes to compare
        self.df_h  = df_h
        # Get int label for column name with 'orig' transform
        self.o_l = np.where(le_t.classes_ == "orig")[0][0]
        
    def _check_bit_flip(self, x: pd.Series, o_l: int, t_l: int, a_s: str, bit_stay: bool):
        #Analyse bit-changes in hash values when using pd.DataFrame.apply()

        if bit_stay: # Bit should stay when comparing orig to transform
            u = self.df_h[
                (self.df_h["filename"] == x["fileA"]) & 
                (self.df_h["transformation"] == o_l)][a_s].values[0]
    
        else: # Bit should flip when comparing transform to transform
            u = self.df_h[
                (self.df_h["filename"] == x["fileA"]) & 
                (self.df_h["transformation"] == t_l)][a_s].values[0]

        v = self.df_h[
            (self.df_h["filename"] == x["fileB"]) & 
            (self.df_h["transformation"] == t_l)][a_s].values[0]

        # Good outcome when a bit should stay
        if bit_stay : good_bits = u == v
        else: good_bits = u != v
        return good_bits

    # Analyse a subset of data
    def fit(self, subset:pd.DataFrame, y_pred:np.ndarray, t_l:int, a_s:str):
        """
        Analyse bits on a given subset of data defined by a triplet.
        The triplet is defined by [algorithm, transform, metric]

        Parameters
        ----------
        subset : pd.DataFrame
            A subset containing the data for the selected triplet
        y_pred : np.ndarray
            Predictions generated for the triplets using MetricMaker
            y_pred = mm.y_pred
        t_l : int
            Integer label for the transform to analyse
        a_s : str
            String for the hashing algorithm to analys

        Returns
        -------
        _type_
            _description_
        """
        self.hashes_bit_length = len(self.df_h.iloc[0][a_s])

        # Select the CM quadrants to analyze bits individually
        FN = subset[(subset["class"] == 1) & (subset["class"] != y_pred)]
        TP = subset[(subset["class"] == 1) & (subset["class"] == y_pred)]
        FP = subset[(subset["class"] == 0) & (subset["class"] != y_pred)]
        TN = subset[(subset["class"] == 0) & (subset["class"] == y_pred)]

        bits = {}
        # Bits should remain for intra analysis
        bits["FN"] = FN.apply(self._check_bit_flip, axis=1, o_l=self.o_l, t_l=t_l, a_s=a_s, bit_stay=True)
        bits["TP"] = TP.apply(self._check_bit_flip, axis=1, o_l=self.o_l, t_l=t_l, a_s=a_s, bit_stay=True)
        # Bit should change for inter analysisself.
        bits["FP"] = FP.apply(self._check_bit_flip, axis=1, o_l=self.o_l, t_l=t_l, a_s=a_s, bit_stay=False)
        bits["TN"] = TN.apply(self._check_bit_flip, axis=1, o_l=self.o_l, t_l=t_l, a_s=a_s, bit_stay=False)

        # Get the size for each part in the CM.
        cm = {"FN": len(FN), "TP": len(TP), "FP": len(FP), "TN": len(TN)}
        
        # Normalize bits but only if there the compnent > 0.
        for component in ["FN", "TP", "FP", "TN"]:
            if cm[f"{component}"] > 0:
                bits[component] = (np.row_stack(bits[component]).sum(axis=0) / cm[f"{component}"])
            else: # if no mistakes, then all bits are cool aka 1.0
                bits[component] = np.repeat(1.0, self.hashes_bit_length)
    
        return pd.DataFrame(bits)

# Create a wrapper parallel metrics
class ComputeMetrics:
    
    def __init__(
        self, 
        le:dict,
        df_d:pd.DataFrame, 
        df_h:pd.DataFrame,
        analyse_bits=False,
        n_jobs=-1,
        backend="loky",
        progress_bar=True,
    ) -> None:
        """
        Compute performance metrics for triplets using JobLib for parallel processing

        Parameters
        ----------
        le : dict
            Dictionary containing LabelEncoders
            Required key values for encoders
        df_d : pd.DataFrame
            Dataframe with distances
        df_h : pd.DataFrame
            Dataframe with hash values
        analyse_bits : bool, optional
            Whether to analyse bit frequency, by default False
        n_jobs : int, optional
            JobLib flag, use all cores, by default -1
        backend : str, optional
            JobLib flag, change backend, by default "loky"
        progress_bar : bool, optional
            Show progress bar using TQDM, by default True
        """
        self.le = le
        self.df_d = df_d
        self.df_h = df_h
        self.analyse_bits = analyse_bits
        self.n_jobs = n_jobs
        self.backend = backend
        self.progress_bar = progress_bar

    def _process_triplet(self, triplet, weighted, normalize=True):
        a_s, t_s, m_s = triplet

        if normalize:
            normalize = "true"
        else:
            normalize = None

        # from string to integer label encoding
        a_l = self.le['a'].transform(np.array(a_s).ravel())[0]
        t_l = self.le['t'].transform(np.array(t_s).ravel())[0]
        m_l = self.le['m'].transform(np.array(m_s).ravel())[0]

        # subset the triplet data
        subset = self.df_d[
            (self.df_d['algo'] == a_l) & 
            (self.df_d['metric'] == m_l)].copy()
        
        y_true = subset['class']
        y_sims = subset[t_s]

        mm = MetricMaker(y_true, y_sims, weighted=weighted)
        tn, fp, fn, tp = mm.get_cm(
            threshold=mm.eer_thresh, 
            normalize=normalize, 
            breakdown=True)

        if self.analyse_bits:
            BA = BitAnalyzer(df_h=self.df_h, le_t=self.le['t'])

            # BA.fit() -> pd.DataFrame
            bits = BA.fit(subset=subset, y_pred=mm.y_pred, t_l=t_l, a_s=a_s)
            return [a_s, t_s, m_s, mm.auc, mm.eer_score, mm.eer_thresh, tn, fp, fn, tp], bits

        else:
            return [a_s, t_s, m_s, mm.auc, mm.eer_score, mm.eer_thresh, tn, fp, fn, tp], None
    
    def fit(self, triplets, weighted=False):
        """
        Fit object on a list of triplets using JobLib

        Parameters
        ----------
        triplets : list
            List of triplets
        weighted : bool, optional
            Apply weighting='balanced' to ConfusionMatrix, by default False

        Returns
        -------
        metrics, bit-weights
            Returns a tuple
        """
        # Use zip() to return tuple (m, b) from process_triplet
        _m, _b = zip(
            *Parallel(
                n_jobs=self.n_jobs, 
                backend=self.backend
                )(delayed(
                    self._process_triplet
                    )(triplet=t, weighted=weighted)
                # TODO: fix the progress bar :/
                for t in tqdm(triplets, desc="Triplet", disable=not self.progress_bar)
            )
        )

        # post-process metrics
        m = np.row_stack(_m)  # type:ignore
        cols = ['Algorithm', 'Transform', 'Metric', 'AUC', 'EER', 'Threshold', 'TN','FP','FN','TP']
        m = pd.DataFrame(m, columns=cols)
        m[m.columns[3:]] = m[m.columns[3:]].astype(float)

        # post-process bitweights
        if self.analyse_bits:
            b = {}
            for i, t in enumerate(triplets):
                t_str = f"{t[0]}_{t[1]}_{t[2]}"
                b[t_str] = _b[i]
        else:
            b = dict()
    
        return m, b

def make_bit_weights(bitfreq:dict, algorithms:list, metrics:list, transforms:list) -> dict:
    """
    Create median bit weights from analysed bit frequency.

    Parameters
    ----------
    bitfreq : dict
        Dictionary containing the bitfrequency dictionary from a the BitAnalyser
    algorithms : list
        Class names from the LabelEncoder used for algorithms
    metrics : list
        Class names from the LabelEncoder used for metrics
    transforms : list
        Class names from the LabelEncoder used for transformations

    Returns
    -------
    dict
        Dictionary with mean weights for each pair (algorithm, metric)
    """
    # Dict to keep weights during loops
    bit_weights = {}

    # Outer loop
    for a in algorithms:
        # Inner loop
        for m in metrics:
            pair = f"{a}_{m}"
            _weights = []

            # Inner-inner loop to process each transform
            for t in transforms:
                # Ignore bits from orig-orig
                if t == 'orig': 
                    continue
                else:
                    # Get the bits from the original triplet
                    freq = bitfreq[f"{a}_{t}_{m}"]
                    result = freq.T.median().values 
                    _weights.append(result)
            bit_weights[pair] = np.median(_weights, axis=0)
            del _weights
    return bit_weights