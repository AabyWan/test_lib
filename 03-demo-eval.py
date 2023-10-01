import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from phaser.utils import load_labelencoders, bin2bool
from phaser.evaluation import ComputeMetrics
from phaser.similarities import IntraDistance, InterDistance

print("Running script.")
script_dir = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
script_dir = f"C:/Users/aabywan/Downloads/Flickr_8k"
# Change to scrip_dir if required.
os.chdir(script_dir)

# Load the label encoders
label_encoders = load_labelencoders(["le_f","le_a","le_t","le_m"], path="./demo_outputs/")
le_f, le_a, le_t, le_m = label_encoders.values()

TRANSFORMS = le_t.classes_
METRICS    = le_m.classes_
ALGORITHMS = le_a.classes_
FIGSIZE    = (5, 3)

print(f"Algorithms available\n{np.column_stack([np.arange(0,len(ALGORITHMS),1), ALGORITHMS])}\n")
print(f"Transformations available\n{np.column_stack([np.arange(0,len(TRANSFORMS),1), TRANSFORMS])}\n")
print(f"Metrics available\n{np.column_stack([np.arange(0,len(METRICS),1), METRICS])}\n")

df_d = pd.read_csv("./demo_outputs/distances.csv.bz2")
df_h = pd.read_csv("./demo_outputs/hashes.csv.bz2")

# convert the strings to arrays
for _a in ALGORITHMS:
    df_h[_a] = df_h[_a].apply(bin2bool)

# Create a label encoder for the class labels
le_c = LabelEncoder()
le_c.classes_ = np.array(['Inter (0)','Intra (1)'])

all_the_bits = {}
evaluation_results = []

triplets = np.array(np.meshgrid(
    ALGORITHMS, 
    TRANSFORMS[:-1], 
    METRICS)).T.reshape(-1,3)

print(f"Number of triplets to analyse: {len(triplets)}")

cm = ComputeMetrics(le_f, le_a, le_t, le_m, df_d, df_h, analyse_bits=True, n_jobs=-1, progress_bar=True)
metrics, bitfreq = cm.fit(triplets=triplets, weighted=False)
print(f"Performance wihtout applying bitweights:")
print(metrics)

# Create mean bitweights
mean_weights = {}

for a in ALGORITHMS:
    for m in METRICS:
        # Aggregate weights across different transforms
        pair = f"{a}_{m}"
        _weights = []

        for t in TRANSFORMS:
            if t == 'orig' : continue
            bits = bitfreq[f"{a}_{t}_{m}"]
            means = bits.T.mean().values 
            _weights.append(means)
        mean_weights[pair] = np.mean(_weights, axis=0)
        del _weights

distance_metrics = {"Hamming": "hamming", "Cosine": "cosine"}

# compute all distances using the new bitweights!
intra = IntraDistance(
    le_t=le_t,
    le_m=le_m,
    le_a=le_a,
    dist_w=mean_weights, # weighted distances!!!
    distance_metrics=distance_metrics,
    set_class=1,
    progress_bar=True)
intra_df = intra.fit(df_h)

from phaser.similarities import find_inter_samplesize
# Compute the inter distances using subsampling
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))

inter = InterDistance(
    le_t,
    le_m,
    le_a,
    dist_w=mean_weights, # weighted distances!!!
    distance_metrics=distance_metrics,
    set_class=0,
    n_samples=n_samples,
    progress_bar=True)
inter_df = inter.fit(df_h)

df_d_w = pd.concat([intra_df, inter_df])

# recompute metrics with weighted distances. No need to analyse bits again !?
cm = ComputeMetrics(le_f, le_a, le_t, le_m, df_d_w, df_h, analyse_bits=False, n_jobs=-1, progress_bar=True)
metrics_w, _ = cm.fit(triplets=triplets, weighted=False)

print(f"Performance WITH applying bitweights:")
print(metrics_w)

# Do a quick plot 
import seaborn as sns
dist_metric='Cosine'
#dist_metric='Hamming'
fig, ax = plt.subplots(1,2,figsize=(8,3), constrained_layout=True, sharex=True, sharey=True)
_ = sns.barplot(
    data=metrics[metrics['Metric'] == dist_metric], #type:ignore
    x='Algorithm', 
    y='AUC', 
    hue='Transform',
    ax=ax[0])

_ = sns.barplot(
    data=metrics_w[metrics_w['Metric'] == dist_metric], #type:ignore
    x='Algorithm', 
    y='AUC', 
    hue='Transform',
    ax=ax[1])

#_ = ax.grid(axis='y', alpha=.25)
_ = ax[0].legend(loc='lower right')
_ = ax[1].legend(loc='lower right')
_ = ax[0].set(title=f"'{dist_metric}' $without$ bit-weighting")
_ = ax[1].set(title=f"'{dist_metric}' $with$ bit-weighting")
fig.savefig("./demo_outputs/03_weight_impact.png")
plt.close()