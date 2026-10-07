import argparse
import os

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

icd_codes = ["Z78.0", "N95.1", "N95.8", "N95.9", "E28.310", "E28.319", "E28.39", "Z79.890"]
hrt_code = "Z79.890"

# (label, ICD-10 prefixes, keywords matched against the diagnosis name)
outcomes = [
    ("Atherosclerosis", ["I70"], ["Atherosclerosis"]),
    ("Alzheimer's disease", ["G30"], ["Alzheimer"]),
    ("Parkinson's disease", ["G20"], ["Parkinson"]),
    ("Rheumatoid arthritis", ["M05", "M06"], ["Rheumatoid arthritis"]),
    ("Heart failure", ["I50"], ["Heart failure"]),
    ("Hypertension", ["I10", "I11", "I12", "I13", "I15", "I16"], ["Hypertension"]),
    ("Ischemic Heart Disease", ["I20", "I21", "I22", "I23", "I24", "I25"], ["Ischemic heart", "Coronary artery", "Myocardial infarction"]),
    ("Stroke", [f"I6{i}" for i in range(10)], ["Stroke", "Cerebral infarction"]),
    ("Type 2 diabetes", ["E11"], ["Type 2 diabetes", "T2DM"]),
    ("Osteoporosis", ["M80", "M81"], ["Osteoporosis"]),
    ("Mental/Behavioral disorders", [f"F{i}" for i in range(10)], ["Mental disorder", "Behavioral disorder", "Depression", "Anxiety"]),
    ("Lupus", ["L93", "M32"], ["Lupus"]),
]

race_map = {
    "White": "White", "White or Caucasian": "White", "Asian": "Asian",
    "Latinx": "Latinx", "Black or African American": "Black",
}

# SFDPH has no clinical notes and no UCSF-derived race field
sites = {
    "ucsf": {"race_col": "ucsfderivedraceethnicity_x", "match_ratio": 4},
    "sfdph": {"race_col": "firstrace", "match_ratio": 8},
}

college_degrees = {"Bachelor", "Bachelor's Degree", "Graduate", "Graduate Degree", "Masters", "Doctoral", "Professional", "College Graduate", "Some Graduate School", "Graduate or Professional Degree"}
private_plans = ["Commercial", "PPO", "HMO", "Blue Cross", "Employer", "Private", "EPO", "POS", "HDHP", "Indemnity"]


def get_args(description, ehr=True, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", required=True, help="folder for intermediate files and results")
    p.add_argument("--site", default="ucsf", choices=list(sites))
    if ehr:
        p.add_argument("--ehr", required=True, help="folder with one subfolder of parquet files per EHR table")
    for args, kwargs in (extra or []):
        p.add_argument(*args, **kwargs)
    a = p.parse_args()
    os.makedirs(a.data, exist_ok=True)
    return a


def connect(folder):
    """DuckDB connection with one view per EHR table."""
    con = duckdb.connect()
    for t in sorted(os.listdir(folder)):
        if os.path.isdir(os.path.join(folder, t)):
            con.sql(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{folder}/{t}/*.parquet')")
    return con


def load(data, name):
    return pd.read_parquet(os.path.join(data, f"{name}.parquet"))


def save(df, data, name):
    df.to_parquet(os.path.join(data, f"{name}.parquet"), index=False)


def years(start, end):
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.days / 365.25


def race(series):
    return series.map(race_map).fillna("Other")


ethnicity_map = {"Hispanic or Latino": "Yes", "Not Hispanic or Latino": "No", "Yes - Hispanic, Latino/a, or Spanish origin": "Yes", "No - Not Hispanic, Latino/a, or Spanish origin": "No"}


def hispanic(value):
    """Looser matching used for the SFDPH model, where ethnicity labels vary."""
    v = str(value).strip().lower()
    if v in ("", "nan", "none") or "unspecified" in v or "decline" in v:
        return None
    if not any(w in v for w in ("hispanic", "latino", "latinx")):
        return None
    return "No" if ("not" in v or "non-" in v or v.startswith("non ")) else "Yes"


def college(series):
    return series.isin(college_degrees).astype(float)


def private_insurance(series):
    return series.apply(lambda x: float(any(v.lower() in str(x).lower() for v in private_plans)) if pd.notna(x) else 0.0)


def postal3(series, top):
    p = series.astype(str).str.extract(r"(\d{3})", expand=False).fillna("Missing")
    common = p[p != "Missing"].value_counts().head(top).index
    return pd.Series(np.where(p == "Missing", "Missing", np.where(p.isin(common), p, "Other")), index=series.index)


def last_bmi(visits, ref, date_col, low=15, high=60, inclusive="both"):
    """Last BMI in range on or before each patient's reference date."""
    v = visits[["patientdurablekey", "bodymassindex", "datekeyvalue"]].copy()
    v["datekeyvalue"] = pd.to_datetime(v["datekeyvalue"], errors="coerce")
    r = ref[["patientdurablekey", date_col]].drop_duplicates("patientdurablekey")
    m = v.merge(r, on="patientdurablekey")
    m = m[(m["datekeyvalue"] <= pd.to_datetime(m[date_col])) & m["bodymassindex"].between(low, high, inclusive=inclusive)]
    m = m.sort_values(["patientdurablekey", "datekeyvalue"], ascending=[True, False])
    return m.groupby("patientdurablekey", as_index=False).first()[["patientdurablekey", "bodymassindex"]].rename(columns={"bodymassindex": "bmi"})


def visits_per_year(enc):
    """Encounters per year of follow-up (minimum one year)."""
    enc = enc.assign(datekeyvalue=pd.to_datetime(enc["datekeyvalue"], errors="coerce"))
    g = enc.groupby("patientdurablekey").agg(n=("encounterkey", "count"), first=("datekeyvalue", "min"), last=("datekeyvalue", "max"))
    return (g["n"] / years(g["first"], g["last"]).clip(lower=1)).rename("visits_per_year").reset_index()


def matches(codes, names, prefixes, keywords):
    by_code = codes.fillna("").str.startswith(tuple(prefixes))
    by_name = names.fillna("").str.lower().str.contains("|".join(k.lower() for k in keywords), regex=True)
    return by_code | by_name


def keep_varying(df, cols):
    """Drop covariates that are constant, under 1% prevalence (binary) or with std under 0.01."""
    keep = []
    for c in cols:
        v = df[c].dropna()
        if v.nunique() == 2 and min(v.mean(), 1 - v.mean()) >= 0.01:
            keep.append(c)
        elif v.nunique() > 2 and v.std() >= 0.01:
            keep.append(c)
    return keep


def fit_cox(df, duration="time", event="event", entry=None):
    """Fit a Cox model, retrying with smaller steps and small penalties if Newton steps fail."""
    err = None
    for penalizer, step in [(0, None), (0, 0.5), (0, 0.2), (1e-4, 0.5), (1e-3, 0.5), (1e-2, 0.5)]:
        try:
            options = {"step_size": step} if step else None
            return CoxPHFitter(penalizer=penalizer).fit(df, duration, event, entry_col=entry, fit_options=options)
        except ValueError as e:
            err = e
    raise err


def hr_row(cph, term):
    lo, hi = np.exp(cph.confidence_intervals_.loc[term].values)
    return {"hazard_ratio": cph.hazard_ratios_[term], "ci_lower": lo, "ci_upper": hi, "p_value": cph.summary.loc[term, "p"]}
