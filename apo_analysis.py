import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from utils import sites, connect, get_args, last_bmi, load, postal3, race, visits_per_year, years

args = get_args("Adverse pregnancy outcomes and age at first menopause-related diagnosis (UCSF only)")
con = connect(args.ehr)
patients, dates, pregnancy = load(args.data, "patients"), load(args.data, "dates"), load(args.data, "pregnancy")

first = patients.drop_duplicates("patientdurablekey")
df = dates.merge(first[["patientdurablekey", "birthdate", sites[args.site]["race_col"]]], on="patientdurablekey")
df["menopause_age"] = years(df["birthdate"], df["onset_date"])
df["race"] = race(df[sites[args.site]["race_col"]])

codes = con.sql("""
    SELECT d.patientdurablekey, t.value AS icd FROM diagnosiseventfact d JOIN diagnosisterminologydim t USING (diagnosiskey)
    WHERE t.type = 'ICD-10-CM' AND d.patientdurablekey IN (SELECT patientdurablekey FROM df)
""").df()
icd = codes["icd"].fillna("")
apos = {"Pre-eclampsia": (icd == "Z87.59") | icd.str.startswith(("O14", "O15")), "Gestational diabetes": icd.str.startswith("O24"), "Gestational hypertension": icd.str.startswith("O13"), "Preterm birth": icd.str.startswith("O60")}
for apo, mask in apos.items():
    df[apo] = df["patientdurablekey"].isin(codes.loc[mask, "patientdurablekey"])
df["n_apo"] = df[list(apos)].sum(axis=1)

# Women with other pregnancy complications (P05.10, O03) are removed from the no-APO reference group
other = codes.loc[icd.str.startswith(("P05.10", "O03")), "patientdurablekey"]
df = df[~((df["n_apo"] == 0) & df["patientdurablekey"].isin(other))]
df = df.merge(pregnancy[["patientdurablekey", "parity", "age_at_first_pregnancy", "years_since_last_birth", "first_pregnancy_date"]], on="patientdurablekey")
print(f"Women with pregnancy records: {len(df):,}")
age = df["menopause_age"]
print(f"Age at first diagnosis: median {age.median():.1f} [{age.quantile(0.25):.1f}-{age.quantile(0.75):.1f}]")
print("Race: " + ", ".join(f"{r} {p:.1%}" for r, p in df["race"].value_counts(normalize=True).items()))
deliveries = con.sql("""
    SELECT f.patientdurablekey, f.lastdeliverydatekeyvalue AS delivery FROM pregnancyfact f
    WHERE f.lastdeliverydatekeyvalue IS NOT NULL AND f.patientdurablekey IN (SELECT patientdurablekey FROM df)
""").df().merge(df[["patientdurablekey", "birthdate"]], on="patientdurablekey")
print(f"Mean age at delivery: {years(deliveries['birthdate'], deliveries['delivery']).groupby(deliveries['patientdurablekey']).mean().mean():.2f}")

# Covariates
df = df.merge(visits_per_year(patients[patients["patientdurablekey"].isin(df["patientdurablekey"])]), on="patientdurablekey", how="left")
visits = load(args.data, "visits")
df = df.merge(last_bmi(visits, dates, "onset_date"), on="patientdurablekey", how="left")
before = visits.merge(df[["patientdurablekey", "first_pregnancy_date"]], on="patientdurablekey")
before = before[pd.to_datetime(before["datekeyvalue"]) < pd.to_datetime(before["first_pregnancy_date"])]
prepreg = before.groupby("patientdurablekey")["bodymassindex"].median().rename("prepreg_bmi")
df = df.merge(prepreg[prepreg.between(15, 60)], on="patientdurablekey", how="left")
obgyn = con.sql("""
    SELECT patientdurablekey, AVG(CASE WHEN providerprimaryspecialty IN
        ('Obstetrics and Gynecology', 'Gynecology', 'Women''s Health', 'Urogynecology', 'Maternal Fetal Medicine') THEN 1 ELSE 0 END) AS obgyn_share
    FROM encounterfact WHERE datekeyvalue IS NOT NULL AND patientdurablekey IN (SELECT patientdurablekey FROM df)
    GROUP BY patientdurablekey
""").df()
df = df.merge(obgyn, on="patientdurablekey", how="left")
df = df.merge(first[["patientdurablekey", "smokingstatus", "highestlevelofeducation", "primaryfinancialclass", "postalcode"]], on="patientdurablekey")
for c in ["visits_per_year", "bmi", "prepreg_bmi", "parity", "age_at_first_pregnancy", "years_since_last_birth", "obgyn_share"]:
    df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
for c in ["race", "highestlevelofeducation", "primaryfinancialclass"]:
    df[c] = df[c].fillna("Missing").astype(str)
df["hrt_post"] = df["hrt_post"].astype(int)
df["ever_smoker"] = df["smokingstatus"].str.lower().str.contains("current|former", na=False).astype(int)
df["postalcode3"] = postal3(df["postalcode"], top=15)
covariates = "C(race) + visits_per_year + bmi + prepreg_bmi + C(postalcode3) + parity + age_at_first_pregnancy + years_since_last_birth + hrt_post + ever_smoker + C(highestlevelofeducation) + C(primaryfinancialclass)"


def adjusted(d, exposed, extra=""):
    m = ols(f"menopause_age ~ exposed + {covariates}{extra}", data=d.assign(exposed=exposed.astype(int))).fit()
    lo, hi = m.conf_int().loc["exposed"]
    return m.params["exposed"], lo, hi, m.pvalues["exposed"]


none = df["n_apo"] == 0
print(f"\nNo APO: n={none.sum():,}, median {df.loc[none, 'menopause_age'].median():.1f}")
rows = []
for apo in apos:
    ages = df.loc[df[apo], "menopause_age"]
    d = df[none | df[apo]]
    b, lo, hi, p = adjusted(d, d[apo])
    rows.append({"apo": apo, "n": len(ages), "median": ages.median(), "mann_whitney_p": stats.mannwhitneyu(ages, df.loc[none, "menopause_age"]).pvalue, "beta": b, "p": p})
by_type = pd.DataFrame(rows)
by_type["p_fdr"] = multipletests(by_type["p"], method="fdr_bh")[1]
print(by_type.to_string(index=False))
print(f"Kruskal-Wallis across groups: p={stats.kruskal(df.loc[none, 'menopause_age'], *[df.loc[df[a], 'menopause_age'] for a in apos]).pvalue:.2e}")

df["any_apo"] = df["n_apo"] > 0
for label, extra in [("Any APO vs none", ""), ("Any APO vs none, adding OB/GYN visit share", " + obgyn_share")]:
    b, lo, hi, p = adjusted(df, df["any_apo"], extra)
    print(f"{label}: beta={b:.2f} years (95% CI {lo:.2f} to {hi:.2f}), p={p:.1e}")

df["burden"] = df["n_apo"].clip(upper=2)
print("\nBurden: " + ", ".join(f"{k}{'+' if k == 2 else ''} APO median {g.median():.1f} [{g.quantile(0.25):.1f}-{g.quantile(0.75):.1f}], n={len(g):,}" for k, g in df.groupby("burden")["menopause_age"]))
tau, p = stats.kendalltau(df["burden"], df["menopause_age"])
print(f"Kendall tau trend test: tau={tau:.2f}, p={p:.1e}")
burden = []
for k in [1, 2]:
    d = df[df["burden"].isin([0, k])]
    burden.append((k,) + adjusted(d, d["burden"] == k))
p_fdr = multipletests([r[4] for r in burden], method="fdr_bh")[1]
for (k, b, lo, hi, p), q in zip(burden, p_fdr):
    print(f"{k}{'+' if k == 2 else ''} vs 0 APOs: beta={b:.2f} years (95% CI {lo:.2f} to {hi:.2f}), p={p:.1e}, FDR p={q:.1e}")
