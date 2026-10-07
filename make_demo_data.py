"""Write a small synthetic EHR dataset with the same tables and columns the pipeline reads. All values are random."""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", default="demo_data")
p.add_argument("--n", type=int, default=6000, help="number of patients")
args = p.parse_args()
rng = np.random.default_rng(0)

codes = {
    "N95.1": "Menopausal and female climacteric states", "Z78.0": "Asymptomatic menopausal state", "Z79.890": "Hormone replacement therapy",
    "E28.310": "Symptomatic premature menopause", "O24.4": "Gestational diabetes", "O13.9": "Gestational hypertension",
    "O60.1": "Preterm labor with preterm delivery", "O14.1": "Severe pre-eclampsia", "O03.9": "Spontaneous abortion",
    "I70.0": "Atherosclerosis of aorta", "G30.9": "Alzheimer's disease", "G20": "Parkinson's disease", "M06.9": "Rheumatoid arthritis",
    "I50.9": "Heart failure", "I10": "Essential hypertension", "I25.10": "Coronary artery disease", "I63.9": "Cerebral infarction",
    "E11.9": "Type 2 diabetes mellitus", "M81.0": "Osteoporosis", "F32.9": "Depression", "F41.1": "Anxiety disorder",
    "F10.20": "Alcohol dependence", "M32.9": "Systemic lupus erythematosus", "J06.9": "Acute upper respiratory infection",
}
code_key = {c: i for i, c in enumerate(codes)}
# yearly risk of each outcome for women, men
risk = {"I70.0": (0.004, 0.003), "G30.9": (0.001, 0.001), "G20": (0.001, 0.002), "M06.9": (0.003, 0.001), "I50.9": (0.002, 0.005), "I10": (0.02, 0.03), "I25.10": (0.006, 0.01), "I63.9": (0.003, 0.004), "E11.9": (0.008, 0.012), "M81.0": (0.012, 0.002), "F32.9": (0.02, 0.012), "F41.1": (0.025, 0.012), "F10.20": (0.003, 0.009), "M32.9": (0.002, 0.0005)}

n = args.n
female = rng.random(n) < 0.6
patients = pd.DataFrame({
    "patientdurablekey": [f"P{i:06d}" for i in range(n)],
    "sex": np.where(female, "Female", "Male"),
    "sexassignedatbirth": np.where(female, "Female", "Male"),
    "birthdate": pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(0, 35 * 365, n), unit="D"),
    "ethnicity": rng.choice(["Hispanic or Latino", "Not Hispanic or Latino", "Unknown/Declined"], n, p=[0.15, 0.8, 0.05]),
    "ucsfderivedraceethnicity_x": rng.choice(["White or Caucasian", "Asian", "Black or African American", "Latinx", "Other"], n, p=[0.45, 0.2, 0.08, 0.1, 0.17]),
    "preferredlanguage": rng.choice(["English", "Spanish", "Cantonese", "Russian"], n, p=[0.85, 0.07, 0.05, 0.03]),
    "postalcode": rng.choice(["941", "949", "940", "945", "946", "950"], n),
    "maritalstatus": rng.choice(["Married", "Single", "Divorced", "Widowed"], n),
    "smokingstatus": rng.choice(["Never Smoker", "Former Smoker", "Current Every Day Smoker", "Unknown"], n, p=[0.6, 0.2, 0.05, 0.15]),
    "highestlevelofeducation": rng.choice(["Bachelor's Degree", "High School", None], n),
    "primaryfinancialclass": rng.choice(["Commercial", "Medicare", "Medi-Cal"], n),
})
patients["firstrace"] = patients["ucsfderivedraceethnicity_x"]

# 25 encounters per patient between 2010 and mid 2025
enc = patients[["patientdurablekey", "birthdate"]].loc[np.repeat(patients.index, 25)].reset_index(drop=True)
enc["encounterkey"] = [f"E{i:07d}" for i in range(len(enc))]
enc["datekeyvalue"] = pd.to_datetime("2010-01-01") + pd.to_timedelta(rng.integers(0, 5650, len(enc)), unit="D")
enc["providerprimaryspecialty"] = rng.choice(["Obstetrics and Gynecology", "Primary Care", "Family Medicine", "Endocrinology"], len(enc))
enc["age"] = (enc["datekeyvalue"] - enc["birthdate"]).dt.days / 365.25

dx = []
# menopause codes for about half of women at an encounter between ages 40 and 60
for key, g in enc[enc["patientdurablekey"].isin(patients.loc[female, "patientdurablekey"])].groupby("patientdurablekey"):
    eligible = g[g["age"].between(40, 60) & (g["datekeyvalue"] >= "2015-01-01")]
    if len(eligible) and rng.random() < 0.5:
        e = eligible.sort_values("datekeyvalue").iloc[rng.integers(len(eligible))]
        for code in rng.choice(["N95.1", "Z78.0", "Z79.890", "E28.310"], rng.integers(1, 3), p=[0.6, 0.2, 0.15, 0.05], replace=False):
            later = eligible[eligible["datekeyvalue"] > e["datekeyvalue"]]
            if code == "Z79.890" and len(later) and rng.random() < 0.5:
                e = later.iloc[0]
            dx.append((key, e["encounterkey"], code, e["datekeyvalue"]))
        if rng.random() < 0.4:
            first = g.sort_values("datekeyvalue").iloc[0]
            for code in rng.choice(["O24.4", "O13.9", "O60.1", "O14.1", "O03.9"], rng.integers(1, 3), replace=False):
                dx.append((key, first["encounterkey"], code, first["datekeyvalue"]))
# outcomes and background codes at random encounters
for code, (rw, rm) in list(risk.items()) + [("J06.9", (0.2, 0.2))]:
    yearly = np.where(enc["patientdurablekey"].isin(patients.loc[female, "patientdurablekey"]), rw, rm) * (1 + (enc["age"] - 40).clip(0) / 20)
    hit = enc[rng.random(len(enc)) < yearly * 15 / 25]
    dx += list(zip(hit["patientdurablekey"], hit["encounterkey"], [code] * len(hit), hit["datekeyvalue"]))
dx = pd.DataFrame(dx, columns=["patientdurablekey", "encounterkey", "code", "startdatekeyvalue"])
dx["diagnosiskey"] = dx["code"].map(code_key)

coded = dx.loc[dx["code"].isin(["N95.1", "Z78.0", "Z79.890", "E28.310"]), ["patientdurablekey", "startdatekeyvalue"]].groupby("patientdurablekey").min()
pregnant = coded.sample(frac=0.4, random_state=0).join(patients.set_index("patientdurablekey")["birthdate"])
delivery = pregnant["birthdate"] + pd.to_timedelta(rng.integers(25 * 365, 38 * 365, len(pregnant)), unit="D")
pregnancy = pd.DataFrame({
    "patientdurablekey": pregnant.index, "pregnancyparacount": rng.integers(1, 4, len(pregnant)), "pregnancygravidacount": rng.integers(1, 5, len(pregnant)),
    "pregnancyestimatedstartdatekeyvalue": delivery - pd.to_timedelta(rng.integers(270, 3650, len(pregnant)), unit="D"), "lastdeliverydatekeyvalue": delivery,
    "lastdeliverygestationalage": rng.integers(33, 41, len(pregnant)),
})

phrases = ["reports a hot flash most days", "wakes with a night sweat", "irregular periods", "vaginal atrophy", "mood swings and irritability", "trouble sleeping", "brain fog", "joint pain", "weight gain", "fatigue", "headaches", "urinary urgency", "low libido"]
women = enc[enc["patientdurablekey"].isin(patients.loc[female, "patientdurablekey"]) & (enc["datekeyvalue"] >= "2012-01-01")]
noted = women.sample(frac=0.3, random_state=0)
text = ["Patient " + ", ".join(rng.choice(phrases, rng.integers(1, 4), replace=False)) + (". Denies headaches." if rng.random() < 0.3 else ".") + (" Discussed menopause." if rng.random() < 0.5 else "") for _ in range(len(noted))]
notes_meta = pd.DataFrame({
    "patientdurablekey": noted["patientdurablekey"].values, "deid_note_key": [f"N{i:07d}" for i in range(len(noted))], "encounter_type": "Office Visit",
    "note_type": "Progress Notes", "encounterkey": noted["encounterkey"].values, "enc_dept_specialty": rng.choice(["Obstetrics and Gynecology", "Primary Care"], len(noted)),
    "deid_service_date": noted["datekeyvalue"].values,
})

tables = {
    "patientdim": patients,
    "encounterfact": enc[["encounterkey", "patientdurablekey", "datekeyvalue", "providerprimaryspecialty"]],
    "diagnosisterminologydim": pd.DataFrame({"diagnosiskey": list(code_key.values()), "type": "ICD-10-CM", "value": list(code_key)}),
    "diagnosisdim": pd.DataFrame({"diagnosiskey": list(code_key.values()), "name": list(codes.values())}),
    "diagnosiseventfact": dx[["encounterkey", "patientdurablekey", "diagnosiskey", "startdatekeyvalue"]],
    "visitfact": enc[["patientdurablekey", "encounterkey"]].assign(bodymassindex=rng.normal(26, 5, len(enc)).round(1)),
    "pregnancyfact": pregnancy,
    "note_metadata": notes_meta,
    "note_text": pd.DataFrame({"deid_note_key": notes_meta["deid_note_key"], "note_text": text}),
}
for name, df in tables.items():
    os.makedirs(os.path.join(args.out, name), exist_ok=True)
    df.to_parquet(os.path.join(args.out, name, "part0.parquet"), index=False)
print(f"Wrote {len(tables)} tables to {args.out} ({n:,} patients, {len(coded):,} with a menopause-related code)")
