import numpy as np

# ===== Patch normalization by mean intensity ========================
def mean_intensity_norm(patch):
	mu = np(np.sum(patch))*1.0/(patch.shape[0]*shape[1])
	return (patch - mu[np.newaxis,np.newaxis])


# ===== Patch normalization by mean intensity ========================
