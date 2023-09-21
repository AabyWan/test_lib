
import pathlib, os, phaser.hashing, phaser.transformers, phaser.similarities._distances
from phaser.utils import ImageLoader as IL
from phaser.utils import dump_labelencoders

# for do hashing
from sklearn.preprocessing import LabelEncoder

# for list modules
from inspect import getmembers, isfunction
import phaser

def list_modular_components():

    # Get hashes - checks each item in phaser.hashing._algorithms and checks to see if the class is a subclass
    # of the abstract class PerceptualHash. If it is, include it in the list of hashes.
    hashes = []
    for name in dir(phaser.hashing):
        try:
            if issubclass(getattr(phaser.hashing, name), phaser.hashing._algorithms.PerceptualHash):
                hashes.append(name)
        except TypeError as err:
            print(err)


    # Get the list of transformers in the same way, except look in phaser.transformers._transforms
    # and check for the phaser.transformers._transforms.Transformer class.
    transformers = []
    for name in dir(phaser.transformers):
        try:
            if issubclass(getattr(phaser.transformers, name), phaser.transformers._transforms.Transformer):
                transformers.append(name)
        except TypeError as err:
            print(err)

    comparison_metrics = [name for name in dir(phaser.similarities._distances) if "_" not in name]

    return {"Hashes": hashes, "Transformers": transformers, "Comparison Metrics": comparison_metrics}

def do_hashing(originals_path:str, algorithms:dict, transformers:list, output_directory:str) -> str:

    # Get list of images
    IMGPATH = originals_path
    list_of_images = [str(i) for i in pathlib.Path(IMGPATH).glob('**/*')]

    ch = phaser.hashing._helpers.ComputeHashes(algorithms, transformers, n_jobs=-1)
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
    outfile = os.path.join(output_directory, "hashes.csv.bz2")
    df.to_csv(outfile, index=False, encoding='utf-8', compression=compression_opts)


if __name__ == "__main__":
    nl = '\n'
    for module_name, functions in list_modular_components().items():
        print( f"{module_name}:{nl}{nl.join(functions)}")
        print(nl)

