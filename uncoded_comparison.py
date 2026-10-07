import os

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from classify_symptoms import classify
from extract_symptoms import symptoms, build_symptom_notes
from utils import connect, get_args, load, save, years

args = get_args("Symptom documentation in age-matched women without a qualifying code (UCSF only)", extra=[
    (["--n-sample"], {"type": int, "default": 3000}),
    (["--seed"], {"type": int, "default": 42}),
    (["--fake-llm"], {"action": "store_true"}),
])
con = connect(args.ehr)
coded = load(args.data, "patients")[["patientdurablekey"]].drop_duplicates()

# Same criteria as the cohort (female, age 35-64 at an encounter in 2015-2024) without the code requirement
eligible = con.sql("""
    SELECT e.patientdurablekey, e.encounterkey, e.datekeyvalue,
        date_diff('year', p.birthdate, e.datekeyvalue::DATE)
        - CASE WHEN month(p.birthdate) > month(e.datekeyvalue::DATE)
                 OR (month(p.birthdate) = month(e.datekeyvalue::DATE) AND day(p.birthdate) > day(e.datekeyvalue::DATE))
               THEN 1 ELSE 0 END AS age
    FROM encounterfact e JOIN patientdim p USING (patientdurablekey)
    WHERE p.sexassignedatbirth != 'Male' AND (p.sexassignedatbirth = 'Female' OR p.sex = 'Female')
      AND e.datekeyvalue::DATE BETWEEN DATE '2015-01-01' AND DATE '2024-12-31'
      AND e.patientdurablekey NOT IN (SELECT patientdurablekey FROM coded)
""").df()
eligible = eligible[eligible["age"].between(35, 64)]
eligible["datekeyvalue"] = pd.to_datetime(eligible["datekeyvalue"])
print(f"Eligible women without a qualifying code: {eligible['patientdurablekey'].nunique():,}")

# Reference date is each woman's last eligible encounter; sample to match the cohort's age at first diagnosis in 1-year bins
ref = eligible.sort_values(["patientdurablekey", "datekeyvalue", "encounterkey"]).groupby("patientdurablekey").tail(1).rename(columns={"datekeyvalue": "reference_date", "age": "reference_age"}).sort_values("patientdurablekey").reset_index(drop=True)
onset = load(args.data, "dates").merge(con.sql("SELECT patientdurablekey, birthdate FROM patientdim").df(), on="patientdurablekey")
onset_age = years(onset["birthdate"], onset["onset_date"])
bins = np.arange(35, 66)
target = np.round(np.histogram(onset_age[onset_age.between(35, 64)], bins=bins)[0] / onset_age.between(35, 64).sum() * args.n_sample).astype(int)
ref["bin"] = pd.cut(ref["reference_age"], bins=bins, right=False, labels=False)
rng = np.random.default_rng(args.seed)
sample = pd.concat([ref[ref["bin"] == b].sample(n=min(t, (ref["bin"] == b).sum()), random_state=rng.integers(1e9)) for b, t in enumerate(target) if t > 0 and (ref["bin"] == b).any()], ignore_index=True)
print(f"Age-matched sample: {len(sample):,}")

encounters = eligible[eligible["patientdurablekey"].isin(sample["patientdurablekey"])][["encounterkey"]].drop_duplicates()
notes_meta = con.sql("""
    SELECT patientdurablekey, deid_note_key, encounter_type, note_type, encounterkey, enc_dept_specialty, deid_service_date
    FROM note_metadata WHERE encounterkey IN (SELECT encounterkey FROM encounters)
""").df().dropna(subset=["patientdurablekey"])
notes = con.sql("SELECT * FROM note_text WHERE deid_note_key IN (SELECT deid_note_key FROM notes_meta)").df()
flagged = build_symptom_notes(notes_meta, notes, sample, "reference_date")
print(f"Notes with at least one symptom: {len(flagged):,}")

labeled = classify(flagged, os.path.join(args.data, "uncoded_llm_parts"), fake=args.fake_llm) if len(flagged) else flagged
save(labeled, args.data, "uncoded_symptom_labels")


def n_with_current(df):
    cols = [f"context_{s}" for s in symptoms if f"context_{s}" in df]
    return df.loc[df[cols].eq("current").any(axis=1), "patientdurablekey"].nunique() if cols else 0


k_uncoded, n_uncoded = n_with_current(labeled), len(sample)
k_coded, n_coded = n_with_current(load(args.data, "symptom_labels")), len(coded)
lo, hi = proportion_confint(k_uncoded, n_uncoded, method="wilson")
z, p = proportions_ztest([k_coded, k_uncoded], [n_coded, n_uncoded])
print(f"Uncoded women with a current symptom: {k_uncoded / n_uncoded:.1%} (95% CI {lo:.1%}-{hi:.1%})")
print(f"Coded cohort: {k_coded / n_coded:.1%}, difference {(k_coded / n_coded - k_uncoded / n_uncoded) * 100:.1f} points, z={z:.1f}, p={p:.1e}")
