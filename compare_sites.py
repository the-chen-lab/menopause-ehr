import argparse

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser(description="Correlation of log hazard ratios between two sites")
p.add_argument("first", help="for example ucsf/cox_results.csv")
p.add_argument("second", help="for example sfdph/cox_results.csv")
args = p.parse_args()

hr = pd.read_csv(args.first, index_col="outcome")[["hazard_ratio"]].join(
    pd.read_csv(args.second, index_col="outcome")[["hazard_ratio"]], lsuffix="_1", rsuffix="_2").dropna()
r, pval = stats.pearsonr(np.log(hr["hazard_ratio_1"]), np.log(hr["hazard_ratio_2"]))
print(f"n={len(hr)}, Pearson r={r:.2f}, p={pval:.1e}")
