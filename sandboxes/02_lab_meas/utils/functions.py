import numpy as np
import pandas as pd


def interp_and_wavg(x: pd.Series):
    if x.count() == 1:
        return x.fillna(0).sum()
    elif x.count() > 1:
        x = x.interpolate('slinear').fillna(
            0)  # Interpolation was failing here
        weights = np.linspace(0, 1, len(x))
        return np.average(x, weights=weights)
    else:
        return np.nan
