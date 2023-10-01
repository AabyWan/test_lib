import pathlib, os
from sklearn.preprocessing import LabelEncoder

from phaser.utils import dump_labelencoders
from phaser.hashing import ComputeHashes, PHASH, ColourHash, PdqHash
from phaser.transformers import Border, Flip

print("Running script.")
script_dir = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
script_dir = f"C:/Users/aabywan/Downloads/Flickr_8k"
# Change to scrip_dir if required.
os.chdir(script_dir)

# Make folder for outputs if not already there.
pathlib.Path("./demo_outputs").mkdir(exist_ok=True)

IMGPATH = os.path.join(script_dir, "Images")

ALGOS = {
    "phash":  PHASH(hash_size=8), 
    "colour": ColourHash(),
    "pdq":    PdqHash()}

TRANS = [
    Border(border_color=(255, 0, 0), border_width=20, saveToPath=""),
    Border(border_color=(255, 0, 0), border_width=30, saveToPath=""),
    Flip(direction="Horizontal", saveToPath="")]

# Prepare for parallel processing
ch = ComputeHashes(ALGOS, TRANS, n_jobs=-1, progress_bar=True)

# Find all the images
list_of_images = [str(i) for i in pathlib.Path(IMGPATH).glob("**/*")]
# Hash all images
df = ch.fit(list_of_images)

# Create label encoders
le = {
    'f': LabelEncoder(),
    't': LabelEncoder(),
    'a': LabelEncoder()}

le['f'].fit(df["filename"])
le['t'].fit(df["transformation"])
le['a'].fit(list(ALGOS.keys()))

# Apply LabelEncoders to data
df["filename"] = le['f'].transform(df["filename"])
df["transformation"] = le['t'].transform(df["transformation"])

# Dump LabelEncoders to disk for use in analysis
dump_labelencoders(le, path="./demo_outputs/")

# Dump the dataset
print(f"{os.getcwd()=}")
compression_opts = dict(method="bz2", compresslevel=9)

df.to_csv(
    "./demo_outputs/hashes.csv.bz2",
    index=False,
    encoding="utf-8",
    compression=compression_opts)

print(f"Script completed")