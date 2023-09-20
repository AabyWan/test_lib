
import pathlib, os, phaser.hashing, phaser.transformers, phaser.similarities._distances
from phaser.utils import ImageLoader as IL
from phaser.utils import dump_labelencoders

# for do hashing
from sklearn.preprocessing import LabelEncoder

# for list modules
from inspect import getmembers, isfunction
import phaser

def list_modules():
    hashes = [name for name in dir(phaser.hashing) if "_" not in name]
    hashes.remove("ComputeHashes")

    transformers = [name for name in dir(phaser.transformers) if "_" not in name]
    comparison_metrics = [name for name in dir(phaser.similarities._distances) if "_" not in name]

    return {"Hashes": hashes, "Transformers": transformers, "Comparison Metrics": comparison_metrics}

def do_hashing(originals_path:str, algorithms:dict, transformers:list, output_directory:str) -> str:

    # Get list of images
    IMGPATH = originals_path
    list_of_images = [str(i) for i in pathlib.Path(IMGPATH).glob('**/*')]

    ch = ComputeHashes(algorithms, transformers, n_jobs=-1)
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
    dump_labelencoders({'le_f':le_f,'le_a':le_a,'le_t':le_t}, path="./demo_outputs/")

    # Dump the dataset
    print(f"{os.getcwd()=}")
    compression_opts = dict(method='bz2', compresslevel=9)
    df.to_csv("./demo_outputs/hashes.csv.bz2", index=False, encoding='utf-8', compression=compression_opts)


if __name__ == "__main__":
    nl = '\n'
    for module_name, functions in list_modules().items():
        print( f"{module_name}:{nl}{nl.join(functions)}")
        print(nl)

