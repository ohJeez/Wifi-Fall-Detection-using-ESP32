import numpy as np


def preprocess(csi):

    csi = csi.squeeze(-1)

    csi = csi.T

    power = np.sum(
        csi ** 2,
        axis=1,
        keepdims=True
    )

    csi = csi ** 2 / (power + 1e-8)

    csi = (
        csi - csi.mean()
    ) / (
        csi.std() + 1e-8
    )

    return csi.astype(np.float32)
