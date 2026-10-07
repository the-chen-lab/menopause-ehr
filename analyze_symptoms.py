import os
from itertools import combinations

import pandas as pd

from extract_symptoms import symptoms
from utils import get_args, load

args = get_args("Symptom prevalence, burden and co-occurrence", ehr=False)
labels = load(args.data, "symptom_labels")
context_cols = [f"context_{s}" for s in symptoms]
labels = labels[labels[context_cols].notna().any(axis=1)]

# Supplementary Table S3: context labels among keyword-flagged notes
table = {}
for s in symptoms:
    flagged = labels[labels[s] == 1]
    context = flagged[f"context_{s}"].astype("string").str.strip().str.lower().fillna("missing")
    table[s] = {"notes": len(flagged), "patients": flagged["patientdurablekey"].nunique(), **context.value_counts(normalize=True).mul(100).round(2)}
table = pd.DataFrame(table).T.fillna(0).astype({"notes": int, "patients": int}).sort_values("patients", ascending=False)
print(table.to_string())

# A patient has a symptom if any note labels it "current"; patients without notes count as having none
current = labels[["patientdurablekey"]].join(labels[context_cols].apply(lambda c: c.str.lower().eq("current")).set_axis(symptoms, axis=1))
by_patient = current.groupby("patientdurablekey")[symptoms].any()
cohort = pd.DataFrame(index=load(args.data, "patients")["patientdurablekey"].unique())
cohort = cohort.join(by_patient.astype(int)).fillna(0).astype(int)
n_symptoms = cohort.sum(axis=1)

prevalence = pd.DataFrame({"n": cohort.sum(), "pct": cohort.mean() * 100}).sort_values("pct", ascending=False)
prevalence.to_csv(os.path.join(args.data, "symptom_prevalence.csv"))
print(f"\nCohort: {len(cohort):,} patients")
print(prevalence.round(1).to_string())
for k in [1, 2, 3, 5]:
    print(f"At least {k} symptom(s): {(n_symptoms >= k).sum():,} ({(n_symptoms >= k).mean():.1%})")

# Supplementary Fig. S2: phi coefficients among patients with labeled notes
phi = by_patient.astype(int).corr()
phi.to_csv(os.path.join(args.data, "symptom_phi.csv"))
pairs = pd.Series({(a, b): phi.loc[a, b] for a, b in combinations(symptoms, 2)}).dropna()
print("\nStrongest co-occurrence: " + ", ".join(f"{a} and {b} ({pairs[a, b]:.3f})" for a, b in pairs.abs().nlargest(5).index))
