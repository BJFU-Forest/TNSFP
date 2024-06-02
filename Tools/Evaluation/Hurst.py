import numpy as np
import pandas as pd


def hurst(vals):
    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)

    N = len(vals)
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

    max_k = int(np.floor(N / 2))
    fit_dict = []
    for k in range(4, max_k + 1):
        RS = []
        # split ts into subsets
        subset_list = [vals[i:i + k] for i in range(0, N, k)]
        if np.mod(N, k) > 0:
            subset_list.pop()

        mean_list = [np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()
            R = max(cumsum_list) - min(cumsum_list)
            S = np.std(subset_list[i])
            RS.append((R + np.spacing(1)) / (S + np.spacing(1)))
        fit_dict.append({"ARS": np.mean(RS), "Tau": k})

    log_ARS = []
    log_Tau = []
    for i in range(len(fit_dict)):
        log_ARS.append(np.log(fit_dict[i]["ARS"]))
        log_Tau.append(np.log(fit_dict[i]["Tau"]))

    Hurst_exponent = np.polyfit(log_Tau, log_ARS, 1)[0]
    return Hurst_exponent