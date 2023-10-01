
## old stuff below for plotting things
# Split into intra and inter for stats
intra_df = df_d[df_d["class"] == 1]
inter_df = df_d[df_d["class"] == 0]

# Create a label encoder for the class labels
le_c = LabelEncoder()
le_c.classes_ = np.array(["Inter (0)", "Intra (1)"])

from phaser.evaluation import macro_stats

for transform in TRANSFORMS[:-1]:
    print(f"\nGenerating macro stats for '{transform}'")
    stats = macro_stats(
        data=intra_df, le_a=le_a, le_m=le_m, transform=transform, style=False
    )
    print(stats.to_latex())

from phaser.plotting import histogram_fig

# Select a specific transform to plot
t_str = "Border_bw30_bc255.0.0"

# Plot for INTER (within images)
fig = histogram_fig(intra_df, le_a, le_m, t_str)
fig.savefig(fname=f"./demo_outputs/FIG_intra_df_{t_str}.png")

# Plot for INTRA (between images)
fig = histogram_fig(inter_df, le_a, le_m, t_str)
fig.savefig(fname=f"./demo_outputs/FIG_inter_df_{t_str}.png")

##################
# MACRO ANALYSIS #
##################

FIGSIZE = (5, 3)
# Select a specific metric and hashing algorithm
m_str = "Hamming"
a_str = "phash"

# Convert to labels
m_label = le_m.transform(np.array(m_str).ravel())
a_label = le_a.transform(np.array(a_str).ravel())

# Subset data
data = df.query(f"algo == {a_label} and metric == {m_label}").copy()

# KDE plot
from phaser.plotting import kde_distributions_ax

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = kde_distributions_ax(
    data, t_str, le_c, fill=True, title=f"{a_str} - {m_str} - {t_str}", ax=ax
)
fig.savefig(fname=f"./demo_outputs/FIG_{a_str}_{m_str}_{t_str}_kde_distributions.png")

# get similarities and true class labels
y_true = data["class"]
y_similarity = data[t_str]

# Prepare metrics for plotting EER and AUC
from phaser.evaluation import MetricMaker

mm = MetricMaker(y_true=y_true, y_similarity=y_similarity, weighted=False)
mm.eer()

# Make predictions and compute cm using EER
cm_eer = mm.get_cm(decision_threshold=mm.eer_thresh, normalize=None)  # type:ignore

# Plot CM using EER threshold

from phaser.plotting import cm_ax

print(f"Plotting CM using EER@{mm.eer_thresh=:.4f} & {mm.eer_score=:.4f}")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = cm_ax(cm=cm_eer, class_labels=le_c.classes_, values_format=".0f", ax=ax)
fig.savefig(
    fname=f"./demo_outputs/FIG_{a_str}_{m_str}_{t_str}_cm_@{mm.eer_thresh:.4f}.png"
)

# Plot EER curve
from phaser.plotting import eer_ax

print(f"Plotting EER curve and finding max FPR threshold")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)

max_fpr = 0.05
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
ax = eer_ax(
    mm.fpr, mm.tpr, mm.thresholds, decision_thresh=mm.eer_thresh, legend=f"", ax=ax
)
fig.savefig(fname=f"./demo_outputs/FIG_{a_str}_{m_str}_{t_str}_EER_curve.png")

# CM using max_fpr
print(f"Plotting CM using {max_fpr=}")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = cm_ax(cm_fpr, class_labels=le_c.classes_, values_format=".0f", ax=ax)
fig.savefig(
    fname=f"./demo_outputs/FIG_{a_str}_{m_str}_{t_str}_cm_@{fpr_threshold:.4f}.png"
)

# ROC curve
from phaser.plotting import roc_ax

print(f"Plotting ROC curve")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
ax = roc_ax(mm.fpr, mm.tpr, roc_auc=mm.auc, legend=f"{a_str}_{m_str}_{t_str}", ax=ax)
fig.savefig(fname=f"./demo_outputs/FIG_{a_str}_{m_str}_{t_str}_ROC_AUC{mm.auc:.4f}.png")
