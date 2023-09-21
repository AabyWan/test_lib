import scipy.spatial.distance as _distance


def hamming(hash1, hash2):
    return _distance.hamming(hash1, hash2)


def cosine(hash1, hash2):
    return _distance.cosine(hash1, hash2)
        