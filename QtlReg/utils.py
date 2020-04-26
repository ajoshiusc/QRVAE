import numpy as np
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter

def make_lesion(img):
    lesion = np.zeros(img.shape)
    mskx, msky = np.where(img >= 0)
    ind = np.random.randint(0, mskx.shape[0])
    mskx = mskx[ind]
    msky = msky[ind]
    #    xyz = np.unravel_index(ind, shape=t1.shape)
    centr = np.array([mskx, msky])[None, :]
    blob, _ = make_blobs(n_samples=10000,
                            n_features=2,
                            centers=centr,
                            cluster_std=np.random.uniform(.1, .5))

    blob = np.int16(
        np.clip(np.round(blob), [0, 0],
                np.array(img.shape) - 1))
    lesion[blob[:, 0], blob[:, 1]] = 1.0
    #lesion[blob.ravel]=1.0

    lesion = gaussian_filter(lesion, 0.5)
    lesion /= lesion.max()

    return lesion

