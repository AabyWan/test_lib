import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from phaser.utils import dump_labelencoders, load_labelencoders, bin2bool
from phaser.similarities import find_inter_samplesize, IntraDistance, InterDistance

print("Running script.")
script_dir = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
script_dir = f"C:/Users/aabywan/Downloads/Flickr_8k"
# Change to scrip_dir if required.
os.chdir(script_dir)

# Read the precomputed hashes
df = pd.read_csv("./demo_outputs/hashes.csv.bz2")

# Load the Label Encoders used when generating hashes
le = load_labelencoders(filename="LabelEncoders", path="./demo_outputs/")
#le_f, le_a, le_t = label_encoders.values()

# Get the unique values and set constants
ALGORITHMS = le['a'].classes_
TRANSFORMS = le['t'].classes_

print(f"{ALGORITHMS=}")
print(f"{TRANSFORMS=}")

# Convert binary hashes to boolean
for a in ALGORITHMS:
    df[a] = df[a].apply(bin2bool)  # type:ignore

# Specify Distance Algorithms
distance_metrics = {"Hamming": "hamming", "Cosine": "cosine"}

# Configure metric LabelEncoder
le['m'] = LabelEncoder().fit(list(distance_metrics.keys()))

# Dump metric LabelEncoder
dump_labelencoders(encoders=le, path="./demo_outputs/")

# Compute the intra distances
intra = IntraDistance(le=le, distance_metrics=distance_metrics,set_class=1,progress_bar=True)
intra_df = intra.fit(df)
print(f"Number of total intra-image comparisons = {len(intra_df)}")

# Compute the inter distances using subsampling
n_samples = find_inter_samplesize(len(df["filename"].unique() * 1))
inter = InterDistance(le,distance_metrics=distance_metrics,set_class=0,n_samples=n_samples,progress_bar=True)
inter_df = inter.fit(df)

print(f"Number of pairwise comparisons = {inter.n_pairs_}")
print(f"Number of inter distances = {len(inter_df)}")

# Combine distances and save to disk
dist_df = pd.concat([intra_df, inter_df])
compression_opts = dict(method="bz2", compresslevel=9)
dist_df.to_csv("./demo_outputs/distances.csv.bz2",index=False,encoding="utf-8",compression=compression_opts,)
print(f"Script completed")