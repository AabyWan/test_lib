
import pathlib, os, phaser.hashing, phaser.transformers, phaser.similarities._distances
from phaser.utils import ImageLoader as IL
from phaser.utils import dump_labelencoders, load_labelencoders, bin2bool

# for do hashing and comparisons
from sklearn.preprocessing import LabelEncoder

# for list modules
from inspect import getmembers, isfunction
import phaser

# for comparisons
import pandas as pd
from phaser.similarities import find_inter_samplesize, IntraDistance, InterDistance


def list_modular_components():

    # Get hashes - checks each item in phaser.hashing._algorithms and checks to see if the class is a subclass
    # of the abstract class PerceptualHash. If it is, include it in the list of hashes.
    hashes = []
    for name in dir(phaser.hashing):
        try:
            if issubclass(getattr(phaser.hashing, name), phaser.hashing._algorithms.PerceptualHash):
                hashes.append(name)
        except TypeError as err:
            pass


    # Get the list of transformers in the same way, except look in phaser.transformers._transforms
    # and check for the phaser.transformers._transforms.Transformer class.
    transformers = []
    for name in dir(phaser.transformers):
        try:
            if issubclass(getattr(phaser.transformers, name), phaser.transformers._transforms.Transformer):
                transformers.append(name)
        except TypeError as err:
            pass

    comparison_metrics = [name for name in dir(phaser.similarities._distances) if "_" not in name]

    return {"Hashes": hashes, "Transformers": transformers, "Comparison Metrics": comparison_metrics}

def do_hashing(originals_path:str, algorithms:dict, transformers:list, output_directory:str, progress_report:bool=True) -> None:

    # Get list of images
    imgpath = originals_path
    list_of_images = [str(i) for i in pathlib.Path(imgpath).glob('**/*')]

    print(f"Found {len(list_of_images)} images in {os.path.abspath(imgpath)}.")
    
    print(f"Creating output directory at {os.path.abspath(output_directory)}...")
    pathlib.Path(output_directory).mkdir(exist_ok=True)

    print("Doing hashing...")
    ch = phaser.hashing._helpers.ComputeHashes(algorithms, transformers, n_jobs=-1, progress_bar=True)
    df = ch.fit(list_of_images)

    # Create label encoders
    le_f = LabelEncoder()
    le_f = le_f.fit(df['filename'])

    le_t = LabelEncoder()
    le_t = le_t.fit(df['transformation'])

    le_a = LabelEncoder()
    le_a = le_a.fit(list(algorithms.keys()))

    # Apply LabelEncoders to data
    df['filename'] = le_f.transform(df['filename'])
    df['transformation'] = le_t.transform(df['transformation'])

    # Dump LabelEncoders to disk for use in analysis
    dump_labelencoders({'le_f':le_f,'le_a':le_a,'le_t':le_t}, path=output_directory)

    # Dump the dataset
    print("Saving hashes.csv and labels for filenames (f), algorithms (a) and transforms (t) to bzip files..")
    compression_opts = dict(method='bz2', compresslevel=9)
    outfile = os.path.join(output_directory, "hashes.csv.bz2")
    df.to_csv(outfile, index=False, encoding='utf-8', compression=compression_opts)

def calcualte_distances(hash_directory:str, progress_report:bool=True) -> None:

    # Read the precomputed hashes from hashes.csv.bz2
    csv_path = os.path.join(hash_directory, "hashes.csv.bz2")

    df = pd.read_csv("./demo_outputs/hashes.csv.bz2")
    print(f"Dataframe loaded from {os.path.abspath(csv_path)}")

    # Load the Label Encoders used when generating hashes
    label_encoders = load_labelencoders(['le_f','le_a','le_t'], path=hash_directory)
    le_f, le_a, le_t = label_encoders.values()

    # Get the unique values and set constants
    ALGORITHMS = le_a.classes_
    TRANSFORMS = le_t.classes_
    print(f"{ALGORITHMS=}")
    print(f"{TRANSFORMS=}")

    # Convert binary hashes to boolean
    for a in ALGORITHMS:
        df[a] = df[a].apply(bin2bool) #type:ignore

    # Define the desired SciPy metrics as string values.
    # TODO make compatible with custom distance functions
    METRICS = ['hamming','cosine']

    # Configure metric LabelEncoder
    le_m = LabelEncoder().fit(METRICS)

    # Dump metric LabelEncoder
    print(f"Saving metric encoder to le_m.")
    dump_labelencoders({'le_m':le_m}, path=hash_directory)

    # Compute the intra distances
    print("\nComputing Intra-distances...")
    intra = IntraDistance(le_t=le_t, le_m=le_m, le_a=le_a, set_class=1, progress_bar=True)
    intra_df = intra.fit(df)
    print(f"Number of total intra-image comparisons = {len(intra_df)}")

    # Compute the inter distances using subsampling
    n_samples = find_inter_samplesize(len(df['filename'].unique()*1))
    print(f"\nComputing Inter-distance with {n_samples} samples per image...")
    inter = InterDistance(le_t, le_m, le_a, set_class=0, n_samples=n_samples, progress_bar=True)
    inter_df = inter.fit(df)

    print(f"Number of pairwise comparisons = {inter.n_pairs_}")
    print(f"Number of inter distances = {len(inter_df)}")

    # Combine distances and save to disk
    dist_df = pd.concat([intra_df,inter_df])
    compression_opts = dict(method='bz2', compresslevel=9)
    distance_path = os.path.join(hash_directory, "distances.csv.bz2")
    print(f"Saving distance scores to {os.path.abspath(distance_path)}.")
    dist_df.to_csv(distance_path, index=False, encoding='utf-8', compression=compression_opts)


def main():
    print("Test listing of modular components...")
    nl = '\n'
    for module_name, functions in list_modular_components().items():
        print( f"{module_name}:{nl}{nl.join(functions)}")
        print(nl)


if __name__ == "__main__":
    main()
