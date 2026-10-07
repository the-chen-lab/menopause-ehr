import re

import pandas as pd

from utils import get_args, load, save

keywords = {
    "hot_flashes": ["hot flash", "hot flush", "hotflash", "hotflush", "heat wave", "heatwave", "sudden warmth", "flushing", "vasomotor", "warm flash", "warm flush", "feeling hot", "sudden heat", "vms", "thermal dysregulation", "heat intolerance", "diaphoresis"],
    "night_sweats": ["night sweat", "nightsweat", "nocturnal sweat", "sleep sweat", "night time sweat", "nighttime sweat", "sweating at night", "wake up sweating", "sweat while sleeping", "drenching sweat", "nocturnal diaphoresis", "sweats at night"],
    "irregular_periods": ["irregular period", "irregular cycle", "irregular menses", "irregular menstrual", "missed period", "skipped period", "skipping period", "cycle change", "menstrual change", "period change", "cycle abnormal", "abnormal period", "erratic period", "unpredictable period", "amenorrhea", "oligomenorrhea", "menstrual irregularity", "menopausal transition"],
    "vaginal_dryness": ["vaginal dry", "vaginal atrophy", "vulvar dry", "vulvovaginal dry", "vaginal lubrication", "lack of lubrication", "atrophic vaginitis", "dyspareunia", "painful intercourse", "painful sex", "discomfort during sex", "vaginal discomfort", "vaginal irritation", "vaginal burn", "gsa", "gsm", "vva", "genitourinary syndrome", "vulvovaginal atrophy", "sexual pain"],
    "mood_changes": ["mood swing", "mood swings", "irritability", "irritable", "anxiety", "anxious", "depression", "depressed", "mood change", "emotional lability", "mood disorder", "weepy", "agitated", "overwhelmed", "emotional instability", "labile mood", "mood lability", "perimenopausal mood"],
    "sleep_issues": ["insomnia", "sleep disturbance", "difficulty sleeping", "trouble sleeping", "sleep problem", "restless sleep", "poor sleep", "sleep disorder", "can't sleep", "cannot sleep", "difficulty falling asleep", "frequent waking", "sleep disruption", "unrefreshing sleep", "sleep fragmentation", "early morning awakening", "middle insomnia"],
    "cognitive_issues": ["brain fog", "memory loss", "memory problem", "forgetful", "forgetfulness", "concentration difficult", "difficulty concentrating", "can't focus", "cognitive decline", "mental fog", "fuzzy thinking", "focus issue", "attention problem", "word finding", "cognitive complaint", "subjective cognitive", "menopausal brain fog"],
    "joint_pain": ["joint pain", "joint ache", "arthralgia", "muscle ache", "stiffness", "joint stiffness", "body ache", "aching joints", "muscle pain", "myalgia", "joint soreness"],
    "weight_changes": ["weight gain", "weight increase", "gained weight", "gaining weight", "abdominal weight", "belly fat", "midsection weight", "metabolism slow", "slow metabolism", "metabolic change", "difficulty losing weight", "hard to lose weight", "can't lose weight", "visceral adiposity", "central adiposity"],
    "fatigue": ["fatigue", "fatigued", "tired", "tiredness", "exhausted", "exhaustion", "low energy", "lack of energy", "no energy", "lethargy", "lethargic", "worn out", "sluggish"],
    "headaches": ["headache", "headaches", "migraine", "migraines", "head pain", "cephalgia", "tension headache"],
    "urinary_issues": ["urinary frequency", "frequent urination", "urinary urgency", "incontinence", "incontinent", "bladder issue", "bladder problem", "leaking urine", "bladder control", "stress incontinence", "urge incontinence"],
    "libido_changes": ["decreased libido", "low libido", "libido", "sex drive", "sexual dysfunction", "loss of interest", "sexual interest", "sexual desire", "hypoactive sexual desire", "sexual avoidance"],
}
symptoms = list(keywords)

context_words = ["menopause", "menopausal", "perimenopause", "perimenopausal", "postmenopausal", "climacteric", "hormone replacement", "hrt", "hormone therapy", "estrogen", "vasomotor", "hot flash", "hot flush", "vms", "progesterone", "progestin", "midlife", "reproductive aging", "ovarian", "follicle", "amenorrhea", "premature ovarian", "menstrual irregularity", "menopausal transition", "reproductive endocrin"]
note_types = ["Progress Notes", "Telephone Encounter", "Patient Instructions", "History and Physical", "Consult", "Office Visit", "ED Provider Notes", "Procedure Note"]
encounter_types = ["Office Visit", "Telephone", "Video Visit", "Hospital Encounter", "Urgent Care", "Procedure Visit"]
specialties = ["Obstetrics and Gynecology", "Primary Care", "General Internal Medicine", "Family Medicine", "Women's Health", "Gynecology", "Endocrinology", "Urogynecology", "Psychiatry", "Rheumatology", "Neurology"]
# any symptom passes in these specialties; the second group also needs a menopause context word
gyn_specialties = ["Obstetrics and Gynecology", "Women's Health", "Gynecology", "Urogynecology", "Reproductive Endocrinology"]
context_specialties = ["Psychiatry", "Rheumatology", "Neurology"]
negations = [r"\bno\b\s+", r"\bdenies?\b\s+", r"\bnegative for\b\s+", r"\bwithout\b\s+", r"\bnot\b\s+", r"\bnever\b\s+", r"\bnone\b\s+", r"doesn't have\s+", r"does not have\s+", r"\bfree of\b\s+", r"\bruled out\b\s+", r"\br/o\b\s+"]
patterns = {s: [re.compile(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])") for k in kws] for s, kws in keywords.items()}


def flag_symptoms(text):
    """1 if a symptom has at least one mention not preceded (within 50 characters) by a negation."""
    text = str(text).lower() if pd.notna(text) else ""
    flags = {}
    for s, regexes in patterns.items():
        flags[s] = int(any(not any(re.search(n, text[max(0, m.start() - 50):m.start()]) for n in negations) for p in regexes for m in p.finditer(text)))
    flags["has_context"] = int(any(w in text for w in context_words))
    return flags


def build_symptom_notes(notes_meta, notes, dates, date_col, window_years=5):
    m = notes_meta.merge(dates[["patientdurablekey", date_col]], on="patientdurablekey")
    service = pd.to_datetime(m["deid_service_date"], errors="coerce")
    ref = pd.to_datetime(m[date_col])
    m = m[(service >= ref - pd.DateOffset(years=window_years)) & (service <= ref + pd.DateOffset(years=window_years))]
    m = m[m["note_type"].isin(note_types) & m["encounter_type"].isin(encounter_types) & m["enc_dept_specialty"].isin(specialties)]
    m = m.drop(columns=date_col).merge(notes, on="deid_note_key", how="left", validate="m:1")
    m = pd.concat([m, pd.DataFrame([flag_symptoms(t) for t in m["note_text"]], index=m.index)], axis=1)

    any_symptom = m[symptoms].sum(axis=1) > 0
    core = m[["hot_flashes", "night_sweats", "irregular_periods", "vaginal_dryness"]].sum(axis=1) > 0
    context = m["has_context"].astype(bool)
    specialty = m["enc_dept_specialty"].isin(gyn_specialties) | (m["enc_dept_specialty"].isin(context_specialties) & context)
    return m[any_symptom & (core | context | specialty)]


if __name__ == "__main__":
    args = get_args("Stage 1 of the NLP pipeline: keyword search of clinical notes", ehr=False, extra=[(["--window-years"], {"type": int, "default": 5})])
    out = build_symptom_notes(load(args.data, "notes_meta"), load(args.data, "notes"), load(args.data, "dates"), "first_code_date", args.window_years)
    save(out, args.data, "symptom_notes")
    print(f"Notes with at least one symptom: {len(out):,} from {out['patientdurablekey'].nunique():,} patients")
