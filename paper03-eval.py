from paper01_conf import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from phaser.utils import load_labelencoders, bin2bool
from phaser.evaluation import ComputeMetrics, make_bit_weights
from phaser.similarities import IntraDistance, InterDistance, find_inter_samplesize
from phaser.plotting import bit_weights_ax

print("Running script.")
# Load the label encoders
le = load_labelencoders(filename="LabelEncoders", path="./demo_outputs/")

# Get values to construct triplets
TRANSFORMS = le["t"].classes_
METRICS = le["m"].classes_
ALGORITHMS = le["a"].classes_

df_d = pd.read_csv("./demo_outputs/distances.csv.bz2")
df_h = pd.read_csv("./demo_outputs/hashes.csv.bz2")

# convert the strings to arrays
for _a in ALGORITHMS:
    df_h[_a] = df_h[_a].apply(bin2bool)

# Generate triplet combinations without 'orig'
triplets = np.array(
    np.meshgrid(ALGORITHMS, [t for t in TRANSFORMS if t != "orig"], METRICS)
).T.reshape(-1, 3)

# Compute metrics for all available triplets
print(f"Number of triplets to analyse: {len(triplets)}")

cm = ComputeMetrics(le, df_d, df_h, analyse_bits=True)
metrics, bitfreq = cm.fit(triplets=triplets, weighted=False)

print(f"Performance without bit weights:")
print(f"Macro AUC={metrics['AUC'].mean():.2f} (±{metrics['AUC'].std():.2f})")
print(f"Macro EER={metrics['EER'].mean():.2f} (±{metrics['EER'].std():.2f})")
print(metrics)

# Plot the bit frequency for each triplet ignoring 'orig'
for triplet in list(bitfreq.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), constrained_layout=True)
    _ = bit_weights_ax(bitfreq[triplet], ax=ax)
    fig.savefig(f"./demo_outputs/figs/03-bit_analysis_{triplet}.png")
    plt.close()

# Create bit_weights (algo,metric)
bit_weights = make_bit_weights(bitfreq, ALGORITHMS, METRICS, TRANSFORMS)

# Plot the applied bitweights for the pairs (algo,metric)
for pair in list(bit_weights.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(6, 1), constrained_layout=True)
    _ = bit_weights_ax(bit_weights[pair].reshape(-1, 1), ax=ax)
    fig.savefig(f"./demo_outputs/figs/03-bit_weights_{pair}.png")
    plt.close()

# Compute all distances using the new bitweights!
intra = IntraDistance(M_DICT, le, 1, bit_weights)
intra_df = intra.fit(df_h)

n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))
inter = InterDistance(M_DICT, le, 0, bit_weights, n_samples=n_samples)
inter_df = inter.fit(df_h)

# Merge the weighted distances
df_d_w = pd.concat([intra_df, inter_df])

# Recompute metrics with weighted distances. No need to analyse bits again !?
cm_w = ComputeMetrics(le, df_d_w, df_h)
metrics_w, _ = cm_w.fit(triplets=triplets, weighted=False)

print(f"Performance with bit weights:")
print(f"Macro AUC={metrics_w['AUC'].mean():.2f} (±{metrics_w['AUC'].std():.2f})")
print(f"Macro EER={metrics_w['EER'].mean():.2f} (±{metrics_w['EER'].std():.2f})")
print(metrics_w)

# Do a quick plot
import seaborn as sns

dist_metric = "Cosine"
# dist_metric='Hamming'
fig, ax = plt.subplots(
    1, 2, figsize=(8, 3), constrained_layout=True, sharex=True, sharey=True
)
_ = sns.barplot(
    data=metrics[metrics["Metric"] == dist_metric],
    x="Algorithm",
    y="AUC",
    hue="Transform",
    ax=ax[0],
)

_ = sns.barplot(
    data=metrics_w[metrics_w["Metric"] == dist_metric],
    x="Algorithm",
    y="AUC",
    hue="Transform",
    ax=ax[1],
)
_ = ax[0].legend(loc="lower right")
_ = ax[1].legend(loc="lower right")
_ = ax[0].set(title=f"'{dist_metric}' $without$ bit-weighting")
_ = ax[1].set(title=f"'{dist_metric}' $with$ bit-weighting")
fig.savefig("./demo_outputs/figs/03_weight_impact.png")
plt.close()
print("Script finished")
