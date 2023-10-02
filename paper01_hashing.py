from paper01_conf import *
from phaser.hashing import PHASH, ColourHash, PdqHash, ComputeHashes
from phaser.transformers import Border, Flip
from phaser.utils import dump_labelencoders

# Configure which hashing algorithms to use and their parameters
ALGOS = {"phash": PHASH(hash_size=8), "colour": ColourHash(), "pdq": PdqHash()}

# Configure the transformations
TRANS = [
    Border(border_color=(255, 0, 0), border_width=20, saveToPath=""),
    Border(border_color=(255, 0, 0), border_width=30, saveToPath=""),
    Flip(direction="Horizontal", saveToPath=""),
]

# Prepare for parallel processing
ch = ComputeHashes(ALGOS, TRANS, n_jobs=-1, progress_bar=True)

# Find all the images and compute hashes
list_of_image_paths = [str(i) for i in pathlib.Path(IMGPATH).glob("**/*")]
df_h = ch.fit(list_of_image_paths)

# Create and fit LabelEncoders according to experiment
le = {
    "f": LabelEncoder().fit(df_h["filename"]),
    "t": LabelEncoder().fit(df_h["transformation"]),
    "a": LabelEncoder().fit(list(ALGOS.keys())),
    "m": LabelEncoder().fit(list(M_DICT.keys())),
    "c": LabelEncoder(),
}

# Hard-code class labels for use when plotting
le["c"].classes_ = np.array(["Inter (0)", "Intra (1)"])

# Apply LabelEncoder on df_h
df_h["filename"] = le["f"].transform(df_h["filename"])
df_h["transformation"] = le["t"].transform(df_h["transformation"])

# Dump LabelEncoders and df_d to disk
dump_labelencoders(le, path="./demo_outputs/")
df_h.to_csv("./demo_outputs/hashes.csv.bz2", index=False)
print(f"Script completed")
