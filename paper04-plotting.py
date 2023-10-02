from paper01_conf import *
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from phaser.utils import load_labelencoders, bin2bool
from phaser.evaluation import dist_stats, MetricMaker
from phaser.plotting import kde_ax, cm_ax, hist_fig, eer_ax, roc_ax

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

## old stuff below for plotting things
# Split into intra and inter for stats
intra_df = df_d[df_d["class"] == 1]
inter_df = df_d[df_d["class"] == 0]

for transform in [t for t in TRANSFORMS if t != "orig"]:
    print(f"\nGenerating macro stats for '{transform}'")
    stats = dist_stats(intra_df, le, transform, style=False)
    print(stats.to_latex())

# Select a specific transform to plot
a_s = "phash"
t_s = "Border_bw30_bc255.0.0"
m_s = "Hamming"

# Convert to labels
a_label = le["a"].transform(np.array(a_s).ravel())
m_label = le["m"].transform(np.array(m_s).ravel())

# Plot for INTER (within images)
fig = hist_fig(intra_df, le, t_s)
fig.savefig(fname=f"./demo_outputs/figs/04-hist_intra_df_{t_s}.png")

# Plot for INTRA (between images)
fig = hist_fig(inter_df, le, t_s)
fig.savefig(fname=f"./demo_outputs/figs/04-hist_inter_df_{t_s}.png")

# Subset data
data = df_d.query(f"algo == {a_label} and metric == {m_label}").copy()

# KDE plot
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = kde_ax(data, t_s, le, fill=True, title=f"{a_s} - {m_s} - {t_s}", ax=ax)
fig.savefig(fname=f"./demo_outputs/figs/04-{a_s}_{m_s}_{t_s}_kde.png")

# get similarities and true class labels
y_true = data["class"]
y_similarity = data[t_s]

# Prepare metrics for plotting EER and AUC
mm = MetricMaker(y_true=y_true, y_similarity=y_similarity, weighted=False)

# Make predictions and compute cm using EER
cm_eer = mm.get_cm(threshold=mm.eer_thresh, normalize=None)  # type:ignore

# Plot CM using EER threshold
print(f"Plotting CM using EER@{mm.eer_thresh=:.4f} & {mm.eer_score=:.4f}")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = cm_ax(cm=cm_eer, class_labels=le["c"].classes_, values_format=".0f", ax=ax)
fig.savefig(f"./demo_outputs/figs/04-{a_s}_{m_s}_{t_s}_cm_@{mm.eer_thresh:.4f}.png")

# Plot EER curve
print(f"Plotting EER curve and finding max FPR threshold")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)

max_fpr = 0.01
fpr_threshold = mm.get_fpr_threshold(max_fpr=max_fpr)
cm_fpr = mm.get_cm(fpr_threshold, normalize="none")
print(f"{max_fpr=} -> {fpr_threshold=:.4f}")
_ = ax.axhline(max_fpr, label=f"FPR={max_fpr:.2f}", color="red")
_ = ax.axvline(
    float(fpr_threshold),
    label=f"FPR={max_fpr:.2f}@{fpr_threshold:.2f}",
    color="red",
    linestyle="--",
)
ax = eer_ax(mm.fpr, mm.tpr, mm.thresholds, threshold=mm.eer_thresh, legend=f"", ax=ax)
fig.savefig(fname=f"./demo_outputs/figs/04-{a_s}_{m_s}_{t_s}_eer.png")

# CM using max_fpr
print(f"Plotting CM using {max_fpr=}")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = cm_ax(cm_fpr, class_labels=le["c"].classes_, values_format=".0f", ax=ax)
fig.savefig(f"./demo_outputs/figs/04-{a_s}_{m_s}_{t_s}_cm_@{fpr_threshold:.4f}.png")

# ROC curve
print(f"Plotting ROC curve")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = roc_ax(mm.fpr, mm.tpr, roc_auc=mm.auc, legend=f"{a_s}_{m_s}_{t_s}", ax=ax)
fig.savefig(fname=f"./demo_outputs/figs/04-{a_s}_{m_s}_{t_s}_ROC_AUC{mm.auc:.4f}.png")
