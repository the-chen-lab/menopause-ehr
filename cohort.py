import pandas as pd

from utils import hrt_code, icd_codes, connect, get_args, save, years

args = get_args("Build the menopause cohort from ICD-10 codes")
con = connect(args.ehr)

# Women with a qualifying code and an encounter at age 35-64 from 2015 onward
codes = con.sql(f"""
    SELECT diagnosiskey, value AS icd_code FROM diagnosisterminologydim
    WHERE type = 'ICD-10-CM' AND value IN {tuple(icd_codes)}
""").df()
dx_all = con.sql("""
    SELECT d.encounterkey, d.patientdurablekey, d.startdatekeyvalue, c.icd_code
    FROM diagnosiseventfact d JOIN codes c USING (diagnosiskey)
""").df()
keys = dx_all[["patientdurablekey"]].dropna().drop_duplicates()

encounters = con.sql("""
    SELECT e.patientdurablekey, e.encounterkey, e.datekeyvalue, p.*,
        date_diff('year', p.birthdate, e.datekeyvalue::DATE)
        - CASE WHEN month(p.birthdate) > month(e.datekeyvalue::DATE)
                 OR (month(p.birthdate) = month(e.datekeyvalue::DATE) AND day(p.birthdate) > day(e.datekeyvalue::DATE))
               THEN 1 ELSE 0 END AS age_at_encounter
    FROM encounterfact e JOIN patientdim p USING (patientdurablekey)
    WHERE e.patientdurablekey IN (SELECT patientdurablekey FROM keys)
      AND p.sexassignedatbirth != 'Male' AND (p.sexassignedatbirth = 'Female' OR p.sex = 'Female')
      AND e.datekeyvalue::DATE >= DATE '2015-01-01'
""").df()
encounters = encounters.loc[:, ~encounters.columns.duplicated()]
encounters = encounters[encounters["age_at_encounter"].between(35, 64)]

dx = dx_all[dx_all["encounterkey"].isin(encounters["encounterkey"])].copy()
dx["startdatekeyvalue"] = pd.to_datetime(dx["startdatekeyvalue"], errors="coerce")
dx = dx[dx["startdatekeyvalue"].between("2015-01-01", "2024-12-31")]
cohort = dx["patientdurablekey"].unique()
patients = encounters[encounters["patientdurablekey"].isin(cohort)]

n_male = con.sql("SELECT COUNT(DISTINCT patientdurablekey) FROM patientdim WHERE sexassignedatbirth = 'Male' AND patientdurablekey IN (SELECT patientdurablekey FROM keys)").fetchone()[0]
print(f"Patients with a menopause-related code: {len(keys):,}")
print(f"Excluded, male: {n_male:,}")
print(f"Excluded, no encounter at age 35-64 since 2015: {len(keys) - n_male - encounters['patientdurablekey'].nunique():,}")
print(f"Excluded, no qualifying code at those encounters (2015-2024): {encounters['patientdurablekey'].nunique() - len(cohort):,}")
print(f"Final cohort: {len(cohort):,}")

# Index date is the first qualifying code excluding Z79.890, unless Z79.890 is the only code.
# The first code of any kind is kept for the notes window and pregnancy intervals.
first_any = dx.groupby("patientdurablekey")["startdatekeyvalue"].min().rename("first_code_date")
onset = dx[dx["icd_code"] != hrt_code].groupby("patientdurablekey")["startdatekeyvalue"].min()
only_hrt = first_any.index.difference(onset.index)
onset = pd.concat([onset, first_any[only_hrt]]).rename("onset_date")
dates = pd.concat([onset, first_any], axis=1).reset_index()

hrt = dx[dx["icd_code"] == hrt_code].merge(dates, on="patientdurablekey")
flags = pd.DataFrame({"patientdurablekey": cohort})
flags["hrt_ever"] = flags["patientdurablekey"].isin(hrt["patientdurablekey"])
flags["hrt_post"] = flags["patientdurablekey"].isin(hrt.loc[hrt["startdatekeyvalue"] > hrt["onset_date"], "patientdurablekey"])
dates = dates.merge(flags, on="patientdurablekey")

print(f"\nOnly qualifying code was Z79.890: {len(only_hrt):,} ({len(only_hrt) / len(cohort):.1%})")
print(f"MHT use (Z79.890), any time: {flags['hrt_ever'].mean():.1%}")
print(f"MHT use, excluding Z79.890-only patients: {flags.loc[~flags['patientdurablekey'].isin(only_hrt), 'hrt_ever'].mean():.1%}")

save(patients, args.data, "patients")
save(dx, args.data, "diagnoses")
save(dates, args.data, "dates")

visits = con.sql("""
    SELECT v.patientdurablekey, v.encounterkey, v.bodymassindex, p.datekeyvalue
    FROM visitfact v JOIN (SELECT DISTINCT encounterkey, datekeyvalue FROM patients) p USING (encounterkey)
""").df()
save(visits, args.data, "visits")

pregnancy = con.sql("""
    SELECT patientdurablekey, MAX(pregnancyparacount) AS parity,
        MIN(pregnancyestimatedstartdatekeyvalue) AS first_pregnancy_date,
        MAX(lastdeliverydatekeyvalue) AS last_delivery_date,
        arg_max(lastdeliverygestationalage, lastdeliverydatekeyvalue) AS last_gestational_age
    FROM pregnancyfact
    WHERE patientdurablekey IN (SELECT patientdurablekey FROM dates) AND lastdeliverydatekeyvalue IS NOT NULL
    GROUP BY patientdurablekey
""").df()
pregnancy = pregnancy.merge(dates[["patientdurablekey", "first_code_date"]], on="patientdurablekey")
pregnancy = pregnancy.merge(patients[["patientdurablekey", "birthdate"]].drop_duplicates("patientdurablekey"), on="patientdurablekey")
pregnancy["age_at_first_pregnancy"] = years(pregnancy["birthdate"], pregnancy["first_pregnancy_date"])
pregnancy["years_since_last_birth"] = years(pregnancy["last_delivery_date"], pregnancy["first_code_date"])
save(pregnancy, args.data, "pregnancy")
print(f"Patients with pregnancy records: {len(pregnancy):,}")

if args.site == "ucsf":
    notes_meta = con.sql("""
        SELECT patientdurablekey, deid_note_key, encounter_type, note_type, encounterkey, enc_dept_specialty, deid_service_date
        FROM note_metadata WHERE encounterkey IN (SELECT encounterkey FROM patients)
    """).df().dropna(subset=["patientdurablekey"])
    notes = con.sql("SELECT * FROM note_text WHERE deid_note_key IN (SELECT deid_note_key FROM notes_meta)").df()
    save(notes_meta, args.data, "notes_meta")
    save(notes, args.data, "notes")
    print(f"Notes: {len(notes):,}")
