from itertools import combinations

import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from utils import ethnicity_map, hrt_code, sites, college, get_args, hispanic, last_bmi, load, private_insurance, race, visits_per_year

args = get_args("Cohort characteristics (Supplementary Table S2) and age at first diagnosis by race and ethnicity", ehr=False)
patients, dx, dates, visits = (load(args.data, f) for f in ["patients", "diagnoses", "dates", "visits"])
patients = patients[patients["age_at_encounter"].notna()]


def first_encounter(codes):
    m = patients.merge(codes[["patientdurablekey", "encounterkey", "startdatekeyvalue"]], on=["patientdurablekey", "encounterkey"])
    return m.sort_values("startdatekeyvalue").groupby("patientdurablekey").first().reset_index()


# Age at the earliest qualifying encounter (Z79.890 only used if it is a patient's only code)
df = first_encounter(dx[dx["icd_code"] != hrt_code])
fallback = first_encounter(dx)
df = pd.concat([df, fallback[~fallback["patientdurablekey"].isin(df["patientdurablekey"])]], ignore_index=True)
df = df.merge(dates, on="patientdurablekey")

age = df["age_at_encounter"]
df["race"] = race(df[sites[args.site]["race_col"]])
df["hispanic"] = df["ethnicity"].map(ethnicity_map)
df = df.merge(visits_per_year(patients), on="patientdurablekey", how="left")
bmi = last_bmi(visits, dates, "onset_date")
df = df.merge(bmi, on="patientdurablekey", how="left")
df["visits_per_year"] = df["visits_per_year"].fillna(df["visits_per_year"].median())
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df["ever_smoker"] = df["smokingstatus"].str.lower().str.contains("current|former", na=False).astype(float)
df["edu_college_plus"] = college(df["highestlevelofeducation"])
df["insurance_private"] = private_insurance(df["primaryfinancialclass"])

n = len(df)
pct = lambda mask: f"{int(mask.sum()):,} ({mask.mean():.1%})"
iqr = lambda s: f"{s.median():.1f} [{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}]"
print(f"Cohort: n={n:,}")
print(f"Age at first diagnosis: mean {age.mean():.1f} (SD {age.std():.1f}), median [IQR] {iqr(age)}")
for lo, hi in [(35, 44), (45, 54), (55, 64)]:
    print(f"  Age {lo}-{hi}: {pct(age.between(lo, hi))}")
for r in ["White", "Asian", "Black", "Latinx", "Other"]:
    print(f"  {r}: {pct(df['race'] == r)}")
print(f"  Hispanic/Latino yes: {pct(df['hispanic'] == 'Yes')}, no: {pct(df['hispanic'] == 'No')}, unknown: {pct(df['hispanic'].isna())}")

languages = {"English": "English", "Spanish": "Spanish", "Cantonese": "Cantonese/Chinese", "Chinese - Cantonese": "Cantonese/Chinese", "Chinese - Mandarin": "Cantonese/Chinese", "Mandarin": "Cantonese/Chinese", "Toishanese": "Cantonese/Chinese"}
marital = {"Married": "Married", "Registered Domestic Partner": "Married", "Single": "Single", "Divorced": "Divorced", "Legally Separated": "Divorced", "RDP-Dissolved": "Divorced", "RDP-LG SEP": "Divorced", "Widowed": "Widowed", "RDP-Widowed": "Widowed", "RDP-Widow": "Widowed"}
smoking = {"Never": "Never", "Never Smoker": "Never", "Passive Smoke Exposure - Never Smoker": "Never", "Former": "Former", "Former Smoker": "Former", "Every Day": "Current", "Current Every Day Smoker": "Current", "Some Days": "Current", "Current Some Day Smoker": "Current", "Light Smoker": "Current", "Light Tobacco Smoker": "Current"}
postal = {c: c for c in ["941", "949", "940", "945"]}
for col, mapping, other in [("preferredlanguage", languages, "Other"), ("maritalstatus", marital, "Other"), ("smokingstatus", smoking, "Unknown/not recorded"), ("postalcode", postal, "Other")]:
    s = df[col].astype(str).str.strip().map(mapping).fillna(other)
    print(f"  {col}: " + ", ".join(f"{v} {pct(s == v)}" for v in list(dict.fromkeys(mapping.values())) + [other]))

table_bmi = last_bmi(visits, dates, "onset_date", 10, 80, "neither")
bmi_vals = table_bmi[table_bmi["patientdurablekey"].isin(df["patientdurablekey"])]["bmi"]
print(f"  BMI: n {pct(df['patientdurablekey'].isin(table_bmi['patientdurablekey']))}, mean {bmi_vals.mean():.1f} (SD {bmi_vals.std():.1f}), median [IQR] {iqr(bmi_vals)}")
enc = patients.assign(day=pd.to_datetime(patients["datekeyvalue"]).dt.normalize()).groupby("patientdurablekey")["day"]
encounter_days = enc.nunique() / ((enc.max() - enc.min()).dt.days / 365.25).clip(lower=1)
print(f"  Encounter-days per year: mean {encounter_days.mean():.1f} (SD {encounter_days.std():.1f}), median [IQR] {iqr(encounter_days)}")
print(f"  MHT any: {pct(df['hrt_ever'])}, post-diagnosis: {pct(df['hrt_post'])}")

covariates = "ever_smoker + edu_college_plus + insurance_private + visits_per_year + bmi"


def compare(d, group):
    """Adjusted p-value for the group term, and how much its coefficient changes without utilization."""
    with_util = ols(f"age_at_encounter ~ C({group}) + {covariates}", data=d).fit()
    without_util = ols(f"age_at_encounter ~ C({group}) + {covariates.replace(' + visits_per_year', '')}", data=d).fit()
    term = [t for t in with_util.params.index if t.startswith(f"C({group})")][0]
    b1, b0 = with_util.params[term], without_util.params[term]
    return with_util.pvalues[term], b1, b0, (b0 - b1) / b0 * 100


h = df.assign(hispanic=df["ethnicity"].map(hispanic)) if args.site == "sfdph" else df
h = h.dropna(subset=["hispanic"])
yes, no = h.loc[h["hispanic"] == "Yes", "age_at_encounter"], h.loc[h["hispanic"] == "No", "age_at_encounter"]
print(f"\nHispanic/Latino: mean {yes.mean():.2f}, median [IQR] {iqr(yes)}")
print(f"Not Hispanic/Latino: mean {no.mean():.2f}, median [IQR] {iqr(no)}")
print(f"  unadjusted t-test p={stats.ttest_ind(yes, no).pvalue:.2e}")
model = ols(f"age_at_encounter ~ C(hispanic) + {covariates}", data=h).fit()
print(f"  adjusted p={anova_lm(model, typ=2).loc['C(hispanic)', 'PR(>F)']:.2e}")
_, b1, b0, share = compare(h, "hispanic")
print(f"  coefficient with utilization {b1:.3f}, without {b0:.3f} ({share:.1f}% attributable to utilization)")

means = df.groupby("race")["age_at_encounter"].mean().sort_values(ascending=False)
print("\nMean age by race: " + ", ".join(f"{r} {m:.2f} (n={(df['race'] == r).sum():,})" for r, m in means.items()))
rows = []
for r1, r2 in combinations(means.index, 2):
    p, b1, b0, share = compare(df[df["race"].isin([r1, r2])], "race")
    rows.append({"pair": f"{r1} vs {r2}", "p": p, "coef_with_util": b1, "coef_without_util": b0, "pct_from_util": share})
pairs = pd.DataFrame(rows)
pairs["p_fdr"] = multipletests(pairs["p"], method="fdr_bh")[1]
print(pairs.round(4).to_string(index=False))
