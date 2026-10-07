import os
import warnings

import numpy as np
import pandas as pd
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests

from utils import outcomes, sites, college, connect, fit_cox, get_args, hr_row, keep_varying, last_bmi, load, matches, postal3, private_insurance, race, years

warnings.filterwarnings("ignore")
args = get_args("Disease risk after first menopause-related diagnosis vs age-matched men")
con = connect(args.ehr)
dates = load(args.data, "dates")
dates = dates[dates["patientdurablekey"].isin(load(args.data, "patients")["patientdurablekey"])]

people = con.sql(f"""
    SELECT patientdurablekey, birthdate, sex, {sites[args.site]['race_col']} AS race_raw, smokingstatus,
           highestlevelofeducation, primaryfinancialclass, postalcode
    FROM patientdim WHERE birthdate IS NOT NULL AND birthdate >= '1900-01-01' AND birthdate <= '2024-12-31'
""").df().drop_duplicates("patientdurablekey")
people["birthdate"] = pd.to_datetime(people["birthdate"])
for r in ["Asian", "Black", "Latinx", "Other"]:
    people[f"race_{r}"] = (race(people["race_raw"]) == r).astype(float)
people["smoker"] = people["smokingstatus"].isin(["Current", "Former", "Yes", "current", "former", "yes"]).astype(float)
people["edu_college_plus"] = college(people["highestlevelofeducation"])
people["insurance_private"] = private_insurance(people["primaryfinancialclass"])
postal = postal3(people["postalcode"], top=15)
postal_cols = [f"postal_{p}" for p in postal.value_counts().index if p not in ("Other", "Missing")]
for c in postal_cols:
    people[c] = (postal == c[7:]).astype(float)

# Match up to N men per woman on birth year (+/- 2 years), each taking the woman's diagnosis date as reference
women = dates[["patientdurablekey", "onset_date"]].merge(people[["patientdurablekey", "birthdate"]], on="patientdurablekey")
women["age"] = years(women["birthdate"], women["onset_date"])
women = women[women["age"].between(35, 64)]
women["birth_year"] = (pd.to_datetime(women["onset_date"]).dt.year - women["age"]).round().astype(int)
men = people[people["sex"] == "Male"]
men_by_year = {y: g for y, g in men.groupby(men["birthdate"].dt.year)}
np.random.seed(42)
matched = []
for year, group in women.groupby("birth_year"):
    pool = [men_by_year[y] for y in range(year - 2, year + 3) if y in men_by_year]
    if not pool:
        continue
    pool = pd.concat(pool, ignore_index=True)
    for _, w in group.iterrows():
        m = pool.sample(n=min(sites[args.site]["match_ratio"], len(pool)))[["patientdurablekey", "birthdate"]].assign(reference_date=w["onset_date"])
        matched.append(m[years(m["birthdate"], m["reference_date"]).between(35, 64)])
men = pd.concat(matched).drop_duplicates("patientdurablekey").assign(is_woman=0)
women = women.rename(columns={"onset_date": "reference_date"})[["patientdurablekey", "birthdate", "reference_date"]].assign(is_woman=1)
ref = pd.concat([women, men], ignore_index=True)
ref["reference_date"] = pd.to_datetime(ref["reference_date"])
print(f"Women: {len(women):,}, matched men: {len(men):,}")

# Covariates
covariates = ["age_at_reference", "reference_year", "bmi", "smoker", "hrt_post", "visits_per_year", "edu_college_plus", "insurance_private", "race_Asian", "race_Black", "race_Latinx", "race_Other"] + postal_cols
ref = ref.merge(people[["patientdurablekey", "smoker", "edu_college_plus", "insurance_private", "race_Asian", "race_Black", "race_Latinx", "race_Other"] + postal_cols], on="patientdurablekey", how="left")
bmi = con.sql("""
    SELECT v.patientdurablekey, v.bodymassindex, e.datekeyvalue FROM visitfact v JOIN encounterfact e USING (encounterkey)
    WHERE v.patientdurablekey IN (SELECT patientdurablekey FROM ref) AND v.bodymassindex > 10 AND v.bodymassindex < 80 AND e.datekeyvalue IS NOT NULL
""").df()
ref = ref.merge(last_bmi(bmi, ref, "reference_date"), on="patientdurablekey", how="left")
ref = ref.merge(dates[["patientdurablekey", "hrt_post"]], on="patientdurablekey", how="left")
enc = con.sql("""
    SELECT patientdurablekey, COUNT(*) AS n, MIN(datekeyvalue::DATE) AS first, MAX(datekeyvalue::DATE) AS last FROM encounterfact
    WHERE datekeyvalue IS NOT NULL AND patientdurablekey IN (SELECT patientdurablekey FROM ref) GROUP BY patientdurablekey
""").df()
enc["visits_per_year"] = enc["n"] / years(enc["first"], enc["last"]).clip(lower=1)
ref = ref.merge(enc[["patientdurablekey", "visits_per_year"]], on="patientdurablekey", how="left")
ref["age_at_reference"] = years(ref["birthdate"], ref["reference_date"])
ref["reference_year"] = ref["reference_date"].dt.year.astype(float)
ref["hrt_post"] = ref["hrt_post"].eq(True).astype(float)
ref[covariates] = ref[covariates].fillna(ref[covariates].median()).fillna(0)

events = con.sql("""
    SELECT d.patientdurablekey, d.startdatekeyvalue, n.name, t.value AS icd
    FROM diagnosiseventfact d
    LEFT JOIN diagnosisdim n USING (diagnosiskey)
    LEFT JOIN (SELECT diagnosiskey, value FROM diagnosisterminologydim WHERE type = 'ICD-10-CM') t USING (diagnosiskey)
    WHERE d.patientdurablekey IN (SELECT patientdurablekey FROM ref)
      AND d.startdatekeyvalue IS NOT NULL AND d.startdatekeyvalue >= '1900-01-01' AND d.startdatekeyvalue <= '2030-12-31'
""").df().merge(ref[["patientdurablekey", "reference_date", "is_woman"]], on="patientdurablekey")
events["years"] = years(events["reference_date"], events["startdatekeyvalue"])
ref = ref.merge(events.groupby("patientdurablekey")["years"].max().rename("last_obs"), on="patientdurablekey", how="left")
ref["last_obs"] = ref["last_obs"].fillna(0)

# Follow-up window for the cumulative prevalence curves uses events within 15 years of the reference date
near = events[events["years"].between(-15, 15)]
ref = ref.merge(near.groupby("patientdurablekey")["years"].agg(follow_start="min", follow_end="max"), on="patientdurablekey", how="left")


def time_to_event(hits, people):
    """Exclude anyone diagnosed in the year before the reference date, then time to first diagnosis or last record (max 15 years)."""
    prevalent = hits.loc[hits["years"].between(-1, 0), "patientdurablekey"]
    hits, people = hits[~hits["patientdurablekey"].isin(prevalent)], people[~people["patientdurablekey"].isin(prevalent)]
    d = people.merge(hits.groupby("patientdurablekey")["years"].min().rename("first_dx"), on="patientdurablekey", how="left")
    d["event"] = d["first_dx"].notna().astype(int)
    d["time"] = d["first_dx"].fillna(d["last_obs"])
    return d[(d["time"] > 0) & (d["time"] <= 15)], hits, people


def women_vs_men(d, covs):
    if d["event"].sum() < 10 or d.groupby("is_woman")["event"].sum().reindex([0, 1], fill_value=0).min() < 3:
        return None
    covs = keep_varying(d, covs)
    data = d[["time", "event", "is_woman"] + covs]
    return fit_cox(data), data


def prevalence_curve(hits, people):
    hits = hits[hits["years"].between(-15, 15)]
    prevalent = hits.loc[hits["years"].between(-1, 0), "patientdurablekey"]
    first_dx = hits[~hits["patientdurablekey"].isin(prevalent)].groupby("patientdurablekey")["years"].min()
    people = people[~people["patientdurablekey"].isin(prevalent)]
    rows = []
    for woman in [1, 0]:
        p = people[people["is_woman"] == woman]
        for start in range(-10, 10):
            followed = p.loc[(p["follow_start"] <= start) & (p["follow_end"] >= start + 1), "patientdurablekey"]
            if len(followed) >= 10:
                rows.append({"cohort": "Women" if woman else "Men", "year": start + 0.5, "n": len(followed), "prevalence": (first_dx.reindex(followed) <= start + 1).mean()})
    return rows


results, windows, subcategories, curves, onset_age = [], [], [], [], []
for label, prefixes, keywords in outcomes:
    hits = events[matches(events["icd"], events["name"], prefixes, keywords)]
    d, hits_incident, ref_incident = time_to_event(hits, ref)
    curves += [{"outcome": label, **r} for r in prevalence_curve(hits, ref)]
    fit = women_vs_men(d, covariates)
    if fit is None:
        print(f"{label}: too few events")
        continue
    cph, data = fit
    ph = proportional_hazard_test(cph, data, time_transform="rank").summary
    r = {"outcome": label, **hr_row(cph, "is_woman"), "n_events": int(d["event"].sum()), "n": len(d), "ph_violated": ", ".join(ph.index[ph["p"] < 0.05])}
    results.append(r)
    print(f"{label}: HR {r['hazard_ratio']:.2f} ({r['ci_lower']:.2f}-{r['ci_upper']:.2f}), proportional hazards violated for: {r['ph_violated'] or 'none'}")

    # Early (0-5 years) and late (5-15 years, landmark at 5) windows
    early = d.assign(event=np.where(d["time"] <= 5, d["event"], 0), time=np.minimum(d["time"], 5))
    late = d[d["time"] >= 5].assign(time=lambda x: x["time"] - 5)
    for name, w in [("early", early), ("late", late)]:
        fit = women_vs_men(w, list(data.columns[3:])) if (w["time"] > 0).all() else None
        if fit is not None:
            windows.append({"outcome": label, "window": name, **hr_row(fit[0], "is_woman"), "n_events": int(w["event"].sum())})

    # Mental/behavioral subcategories
    if label == "Mental/Behavioral disorders":
        for sub, sub_prefixes in [("Substance use (F10-F19)", ["F1"]), ("Mood (F30-F39)", ["F3"]), ("Anxiety/stress-related (F40-F48)", ["F40", "F41", "F42", "F43", "F44", "F45", "F48"])]:
            sub_hits = hits_incident[hits_incident["icd"].fillna("").str.startswith(tuple(sub_prefixes))]
            fit = women_vs_men(time_to_event(sub_hits, ref_incident)[0], covariates)
            if fit is not None:
                subcategories.append({"subcategory": sub, **hr_row(fit[0], "is_woman")})

    # Within women: age as the time scale, entering at age at diagnosis, so the HR is per year later diagnosis
    w = time_to_event(hits[hits["is_woman"] == 1], ref[ref["is_woman"] == 1])[0]
    if w["event"].sum() >= 10:
        covs = keep_varying(w, covariates[1:])
        w = w.assign(entry=w["age_at_reference"], exit=w["age_at_reference"] + w["time"])
        cph = fit_cox(w[["entry", "exit", "event", "age_at_reference"] + covs], duration="exit", entry="entry")
        onset_age.append({"outcome": label, **hr_row(cph, "age_at_reference"), "n_events": int(w["event"].sum())})

for rows, name in [(results, "cox_results"), (subcategories, "cox_mental_subcategories"), (onset_age, "cox_onset_age")]:
    df = pd.DataFrame(rows)
    df["p_fdr"] = multipletests(df["p_value"], method="fdr_bh")[1]
    df.to_csv(os.path.join(args.data, f"{name}.csv"), index=False)
    print(f"\n{name}\n{df.drop(columns='ph_violated', errors='ignore').round(4).to_string(index=False)}")
windows = pd.DataFrame(windows)
windows["p_fdr"] = windows.groupby("window")["p_value"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
windows.to_csv(os.path.join(args.data, "cox_windows.csv"), index=False)
print(f"\ncox_windows\n{windows.round(4).to_string(index=False)}")
pd.DataFrame(curves).to_csv(os.path.join(args.data, "cumulative_prevalence.csv"), index=False)
