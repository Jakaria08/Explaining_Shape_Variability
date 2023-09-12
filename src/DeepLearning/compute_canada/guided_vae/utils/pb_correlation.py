import numpy as np
import math

def point_biserial_correlation(z_batch, y_batch):
    # Extract data for the two categories
    z_1 = z_batch[y_batch.flatten() == 1.0]
    z_0 = z_batch[y_batch.flatten() == 0.0]
    n_1 = len(z_1)
    n_0 = len(z_0)
    n = n_1 + n_0

    # Calculate means for the two categories
    mean_z_1 = np.mean(z_1[:, 0])
    mean_z_0 = np.mean(z_0[:, 0])

    # Multiplier
    mlt = math.sqrt((n_1 * n_0) / (n**2))

    # Calculate point biserial correlation
    r_pb = (mean_z_1 - mean_z_0) / (np.std(z_batch[:, 0]) * mlt)

    return r_pb