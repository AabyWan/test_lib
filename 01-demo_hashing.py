import pathlib, os

pathlib.Path("./demo_outputs").mkdir(exist_ok=True)

from phaser.utils import ImageLoader as IL
from phaser.utils import dump_labelencoders
from phaser.hashing import PHASH

print("Running script.")
script_dir = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
print(script_dir)

IMGPATH = os.path.join(script_dir, "images")

list_of_images = [str(i) for i in pathlib.Path(IMGPATH).glob("**/*")]

from phaser.hashing import ComputeHashes, PHASH, ColourHash

algorithms = {"phash": PHASH(hash_size=8, highfreq_factor=4), "colour": ColourHash()}

from phaser.transformers import Border, Flip

transformers = [
    Border(border_color=(255, 0, 0), border_width=30, saveToPath=""),
    Flip(direction="Horizontal", saveToPath=""),
]

ch = ComputeHashes(algorithms, transformers, n_jobs=-1, progress_bar=True)
df = ch.fit(list_of_images)

from sklearn.preprocessing import LabelEncoder

# Create label encoders
le_f = LabelEncoder()
le_f = le_f.fit(df["filename"])

le_t = LabelEncoder()
le_t = le_t.fit(df["transformation"])

le_a = LabelEncoder()
le_a = le_a.fit(list(algorithms.keys()))

# Apply LabelEncoders to data
df["filename"] = le_f.transform(df["filename"])
df["transformation"] = le_t.transform(df["transformation"])

# Dump LabelEncoders to disk for use in analysis
dump_labelencoders({"le_f": le_f, "le_a": le_a, "le_t": le_t}, path="./demo_outputs/")

# Dump the dataset
print(f"{os.getcwd()=}")
compression_opts = dict(method="bz2", compresslevel=9)
df.to_csv(
    "./demo_outputs/hashes.csv.bz2",
    index=False,
    encoding="utf-8",
    compression=compression_opts,
)
