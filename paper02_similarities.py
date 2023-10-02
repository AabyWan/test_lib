from paper01_conf import *
from phaser.utils import load_labelencoders, bin2bool
from phaser.similarities import IntraDistance, find_inter_samplesize, InterDistance

# Read the precomputed hashes
df_h = pd.read_csv("./demo_outputs/hashes.csv.bz2")

# Load the Label Encoders used when generating hashes
le = load_labelencoders(filename="LabelEncoders", path="./demo_outputs/")

# Convert binary hashes to boolean for distance computation
for a in le["a"].classes_:
    df_h[a] = df_h[a].apply(bin2bool)

# Compute the intra distances
intra = IntraDistance(M_DICT, le, 1, progress_bar=True)
intra_df = intra.fit(df_h)
print(f"Number of total intra-image comparisons = {len(intra_df)}")

# Compute the inter distances using subsampling
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))
inter = InterDistance(M_DICT, le, set_class=0, n_samples=n_samples, progress_bar=True)
inter_df = inter.fit(df_h)

print(f"Number of pairwise comparisons = {inter.n_pairs_}")
print(f"Number of total inter distances = {len(inter_df)}")

# Combine distances and save to disk
df_d = pd.concat([intra_df, inter_df])
df_d.to_csv("./demo_outputs/distances.csv.bz2", index=False)
print(f"Script completed")