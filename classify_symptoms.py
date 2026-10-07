import glob
import json
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from extract_symptoms import symptoms
from utils import get_args, load, save

context_labels = ["current", "past", "family_history", "concern", "negated", "uncertain", "not_mentioned"]

# The prompt text below is exactly what was used in the paper
descriptions = {
    "hot_flashes": "sudden heat sensations, flushing, vasomotor symptoms (VMS), diaphoresis (profuse sweating due to heat surge), thermal dysregulation, heat intolerance",
    "night_sweats": "sweating during sleep, nocturnal diaphoresis, drenching sweats at night, nighttime sweating that disrupts sleep",
    "irregular_periods": "irregular, missed, skipped, or erratic menstrual cycles; amenorrhea; oligomenorrhea; menopausal transition; menstrual irregularity",
    "vaginal_dryness": "vaginal dryness or atrophy, vulvovaginal atrophy (VVA), genitourinary syndrome of menopause (GSM/GSA), atrophic vaginitis, dyspareunia, painful intercourse, lack of vaginal lubrication, vaginal irritation or burning",
    "mood_changes": "mood swings, irritability, anxiety, depression, emotional lability, mood disorder, weepiness, agitation, feeling overwhelmed — in the context of the patient's own current emotional state, NOT a longstanding psychiatric diagnosis unrelated to menopause",
    "sleep_issues": "insomnia, difficulty sleeping, sleep disturbance, poor sleep quality, restless sleep, frequent waking, sleep fragmentation, early morning awakening, unrefreshing sleep — NOT incidental waking to urinate unless sleep disruption is also noted",
    "cognitive_issues": "brain fog, memory problems or loss, forgetfulness, difficulty concentrating, subjective cognitive decline, word-finding difficulty, mental fog — as reported by the patient, not an objective cognitive test result",
    "joint_pain": "joint pain, arthralgia, joint aches or stiffness, body aches, muscle pain, myalgia, aching joints — NOT post-surgical or focal injury pain unless clearly generalized and musculoskeletal",
    "weight_changes": "weight gain, difficulty losing weight, abdominal or visceral adiposity, belly fat, metabolic changes, slow metabolism — in the context of menopausal or perimenopausal weight change",
    "fatigue": "fatigue, exhaustion, low energy, lethargy, tiredness, sluggishness — as a symptom the patient is experiencing, NOT fatigue as an expected side effect of a medication or acute illness",
    "headaches": "headache, migraine, tension headache, head pain, cephalgia — as a symptom the patient currently experiences or reports",
    "urinary_issues": "urinary frequency, urinary urgency, urinary incontinence, bladder control problems, leaking urine, stress or urge incontinence — NOT a urinary tract infection unless incontinence is also documented",
    "libido_changes": "decreased libido, low sex drive, sexual dysfunction, hypoactive sexual desire disorder, loss of sexual interest — as reported by the patient",
}

prompt_template = """You are a clinical NLP system extracting structured data from EHR notes.

Your task: for each menopause-related symptom listed below, read the clinical note
and classify the CONTEXT in which that symptom appears FOR THE INDEX PATIENT ONLY
(not relatives or other patients).

Each symptom entry includes a description of the specific clinical terms and
concepts you should look for in the note.

---
SYMPTOMS TO CLASSIFY (name: what to look for):
{symptoms}

---
CONTEXT LABELS — choose exactly one per symptom:
  - current:        the patient is actively experiencing this symptom now or very recently
  - past:           the patient experienced this in the past and it is now resolved
  - family_history: mentioned only for a family member, not the patient
  - concern:        patient or clinician worried the patient may develop this, but not currently experiencing it
  - negated:        explicitly documented as absent (e.g. "denies X", "no X", "negative for X")
  - uncertain:      mentioned but context is genuinely ambiguous
  - not_mentioned:  the regex keyword was a false positive; this symptom is not truly present in the note

SEVERITY LABELS — choose exactly one per symptom:
  - mild, moderate, severe, not_specified
  Use not_specified for: negated, not_mentioned, family_history, concern.

---
Return ONLY a JSON object in this exact format.
No explanation, no markdown fences, no extra keys:
{{
  "symptoms": [
    {{"name": "<symptom_name>", "context": "<context_label>", "severity": "<severity_label>"}}
  ]
}}

You MUST include every symptom in the list, even if its context is not_mentioned.

---
CLINICAL NOTE:
\"\"\"{note}\"\"\""""


def parse(raw):
    text = raw.replace("\u202f", " ").replace("\u00a0", " ").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    try:
        items = json.loads(text)["symptoms"]
        return {i["name"]: i["context"] for i in items if isinstance(i, dict) and "name" in i}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def call_llm(system, prompt):
    """Return the LLM's text response. Edit this to call the LLM API of your choice (the paper used gpt-oss-20b with temperature 0)."""
    raise NotImplementedError("Edit call_llm in classify_symptoms.py to call your LLM")


def classify_note(row, fake):
    present = [s for s in symptoms if row.get(s) == 1]
    if fake:
        rng = np.random.default_rng(zlib.crc32(str(row["deid_note_key"]).encode()))
        labels = {s: rng.choice(context_labels, p=[0.5, 0.1, 0.01, 0.01, 0.1, 0.03, 0.25]) for s in present}
        return {f"context_{s}": labels.get(s) for s in symptoms}
    block = "\n".join(f"  - {s}: {descriptions[s]}" for s in present)
    prompt = prompt_template.format(symptoms=block, note=row["note_text"] or "")
    try:
        labels = parse(call_llm("You are a helpful clinical NLP assistant.", prompt))
    except NotImplementedError:
        raise
    except Exception as e:
        print(f"note {row['deid_note_key']}: {e}")
        labels = {}
    return {f"context_{s}": labels.get(s) if s in present else None for s in symptoms}


def classify(notes, parts_dir, workers=16, chunk=500, fake=False):
    """Label each flagged symptom in each note. Results are written in chunks so an interrupted run can resume."""
    os.makedirs(parts_dir, exist_ok=True)
    done = glob.glob(os.path.join(parts_dir, "*.parquet"))
    if done:
        finished = pd.concat([pd.read_parquet(f, columns=["deid_note_key"]) for f in done])["deid_note_key"]
        notes = notes[~notes["deid_note_key"].isin(finished)]
    rows = notes[["patientdurablekey", "deid_note_key", "deid_service_date", "note_text"] + symptoms].to_dict("records")
    with ThreadPoolExecutor(workers) as pool:
        for start in tqdm(range(0, len(rows), chunk)):
            batch = rows[start:start + chunk]
            labels = list(pool.map(lambda r: classify_note(r, fake), batch))
            out = pd.DataFrame(batch).join(pd.DataFrame(labels))
            out.to_parquet(os.path.join(parts_dir, f"{len(done) + start // chunk:06d}.parquet"), index=False)
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))], ignore_index=True)


if __name__ == "__main__":
    args = get_args("Stage 2 of the NLP pipeline: LLM context classification of flagged symptoms", ehr=False, extra=[
        (["--workers"], {"type": int, "default": 16}),
        (["--fake-llm"], {"action": "store_true", "help": "assign random labels instead of calling a model (demo only)"}),
    ])
    labeled = classify(load(args.data, "symptom_notes"), os.path.join(args.data, "llm_parts"), args.workers, fake=args.fake_llm)
    save(labeled, args.data, "symptom_labels")
    print(f"Labeled {len(labeled):,} notes")
