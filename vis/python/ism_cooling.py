# ism_cooling.py

import numpy as np

def ISMCoolFn(temp):
    """
    Vectorized ISM cooling function.
    `temp` can be a scalar, NumPy array (any shape), or something convertible to an array.
    Returns an array of the same shape as `temp`.
    """

    temp = np.asarray(temp, dtype=float)

    lhd = np.array([
        -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
        -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
        -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
        -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
        -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
        -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
        -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
        -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
        -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
        -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
        -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
        -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
        -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928
    ])

    logt = np.log10(temp)

    # Initialize result array
    Lambda = np.zeros_like(logt, dtype=float)

    # Region 1: logT <= 4.2 -> 0 Lambda(10^4.2)
    mask_low = logt <= 4.2
    logt_4p2 = 4.2
    ipps_low = int(25.0 * logt_4p2) - 103
    ipps_low = np.clip(ipps_low, 0, 100)
    x0_low = 4.12 + 0.04 * ipps_low
    dx_low = logt_4p2 - x0_low
    logcool_4p2 = (lhd[ipps_low + 1] * dx_low - lhd[ipps_low] * (dx_low - 0.04)) * 25.0
    Lambda_low = 10.0 ** logcool_4p2
    Lambda[mask_low] = Lambda_low
    
    # Region 2: logT > 8.15 -> CGOLS fit
    mask_hi = logt > 8.15
    Lambda[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # Region 3: 4.2 < logT <= 8.15 -> table + interpolation
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        logt_mid = logt[mask_mid]

        ipps = (25.0 * logt_mid).astype(int) - 103
        ipps = np.clip(ipps, 0, 100)

        x0 = 4.12 + 0.04 * ipps.astype(float)
        dx = logt_mid - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        Lambda[mask_mid] = 10.0 ** logcool

    return Lambda
