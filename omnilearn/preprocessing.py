import numpy as np


def revert_npart(npart, name="30"):
    # Reverse the preprocessing to recover the particle multiplicity
    stats = {
        "30": (29.03636, 2.7629626),
        "49": (21.66242333, 8.86935969),
        "150": (49.398304, 20.772636),
        "279": (57.28675, 29.41252836),
    }
    mean, std = stats[name]
    return np.round(npart * std + mean).astype(np.int32)

