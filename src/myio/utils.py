from __future__ import annotations

import numpy as np


def dbfs_to_rms(dbfs: float) -> float:
    """Convert an RMS-referenced dBFS level to linear RMS amplitude.

    dBFS is a 20*log10 amplitude ratio, so the inverse is 10^(db/20).
    """
    return 10.0 ** (dbfs / 20.0)


def rms_to_dbfs(rms: float) -> float:
    """Convert linear RMS amplitude to dBFS (RMS-referenced).

    Returns -inf for rms=0 (silence) and NaN for rms<0 (undefined).
    """
    return 20.0 * float(np.log10(rms))
