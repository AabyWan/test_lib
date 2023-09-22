import pandas as pd
from sklearn.preprocessing import LabelEncoder
from phaser.utils import dump_labelencoders, load_labelencoders, bin2bool

# Read the precomputed hashes
df = pd.read_csv("./demo_outputs/hashes.csv.bz2")

# Load the Label Encoders used when generating hashes
label_encoders = load_labelencoders(["le_f", "le_a", "le_t"], path="./demo_outputs/")
le_f, le_a, le_t = label_encoders.values()

# Get the unique values and set constants
ALGORITHMS = le_a.classes_
TRANSFORMS = le_t.classes_
print(f"{ALGORITHMS=}")
print(f"{TRANSFORMS=}")

# Convert binary hashes to boolean
for a in ALGORITHMS:
    df[a] = df[a].apply(bin2bool)  # type:ignore

from phaser.similarities import test_synthetic

# Specify Distance Algorithms
# If the distance metric is part of scipy.spatial.distance, specify the name as a string
# If a custom distance metric is provided in phaser.similarities._distances.py, import and pass the function reference
distance_metrics = {
    "Hamming": "hamming",
    "Cosine": "cosine",
    "Test_Synthetic": test_synthetic,
}

# Configure metric LabelEncoder
le_m = LabelEncoder().fit(list(distance_metrics.keys()))

# Dump metric LabelEncoder
dump_labelencoders({"le_m": le_m}, path="./demo_outputs/")

# Compute the intra distances
from phaser.similarities import find_inter_samplesize, IntraDistance, InterDistance

intra = IntraDistance(
    le_t=le_t,
    le_m=le_m,
    le_a=le_a,
    distance_metrics=distance_metrics,
    set_class=1,
    progress_bar=True,
)
intra_df = intra.fit(df)
print(f"Number of total intra-image comparisons = {len(intra_df)}")

# Compute the inter distances using subsampling
n_samples = find_inter_samplesize(len(df["filename"].unique() * 1))
inter = InterDistance(
    le_t,
    le_m,
    le_a,
    distance_metrics=distance_metrics,
    set_class=0,
    n_samples=n_samples,
    progress_bar=True,
)
inter_df = inter.fit(df)

print(f"Number of pairwise comparisons = {inter.n_pairs_}")
print(f"Number of inter distances = {len(inter_df)}")

# Combine distances and save to disk
dist_df = pd.concat([intra_df, inter_df])
compression_opts = dict(method="bz2", compresslevel=9)
dist_df.to_csv(
    "./demo_outputs/distances.csv.bz2",
    index=False,
    encoding="utf-8",
    compression=compression_opts,
)
