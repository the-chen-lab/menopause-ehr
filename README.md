# Menopause EHR

This is the code for reproducing results from the paper: "Characterization of menopause onset and associated disease risks using large-scale electronic health records"

We identify a menopause cohort from ICD-10 codes in two health systems (UCSF Health and the San Francisco Department of Public Health), extract menopause symptoms from clinical notes with a two-stage keyword and LLM pipeline, relate adverse pregnancy outcomes to age at first menopause-related diagnosis, and compare disease risk after diagnosis to age-matched men.

The EHR data used in the paper cannot be shared (see Data availability in the paper), so we include a script that generates a small synthetic dataset with the same tables and columns. All results on the synthetic data are meaningless and only show that the code runs.

## System requirements
The code is written in Python and needs no special hardware. We tested it on macOS with Python 3.13.2, and ran the analyses in the paper on a Linux server with Python 3.13.5 and 96 GB of memory available to DuckDB. Package versions are in `requirements.txt`.

The symptom classification step (`classify_symptoms.py`) calls an LLM of your choice. We used the open-weight model gpt-oss-20b. Every other step runs locally on a CPU.

## Setup
To run our code, you will need to set up a new environment. To do so, run the following commands (which should take around one minute):
```bash
git clone https://github.com/the-chen-lab/menopause-ehr.git
cd menopause-ehr
conda create -n menopause_ehr python=3.13
conda activate menopause_ehr
pip install -r requirements.txt
```

## Demo
The commands below build a synthetic dataset with 6,000 patients and run the full pipeline for both sites. `--fake-llm` replaces the model call with random labels so the demo runs without an LLM. The whole demo takes under a minute on a laptop.
```bash
python make_demo_data.py

python cohort.py --ehr demo_data --data results/ucsf
python demographics.py --data results/ucsf
python extract_symptoms.py --data results/ucsf
python classify_symptoms.py --data results/ucsf --fake-llm
python analyze_symptoms.py --data results/ucsf
python uncoded_comparison.py --ehr demo_data --data results/ucsf --fake-llm
python apo_analysis.py --ehr demo_data --data results/ucsf
python survival.py --ehr demo_data --data results/ucsf

python cohort.py --site sfdph --ehr demo_data --data results/sfdph
python demographics.py --site sfdph --data results/sfdph
python survival.py --site sfdph --ehr demo_data --data results/sfdph
python compare_sites.py results/ucsf/cox_results.csv results/sfdph/cox_results.csv
```
Each script prints its results. Intermediate tables are saved as parquet files in the `--data` folder, and the numbers behind each figure are saved as csv files there too (for example `cox_results.csv` and `cumulative_prevalence.csv`). The demo should report a cohort of 1,408 women and, for the UCSF site, an osteoporosis hazard ratio of about 9 for women versus matched men.

## Running on your own data
Point `--ehr` at a folder with one subfolder of parquet files per table, named as in the UCSF and SFDPH de-identified EHR databases: `patientdim`, `encounterfact`, `diagnosiseventfact`, `diagnosisdim`, `diagnosisterminologydim`, `visitfact`, `pregnancyfact`, `note_metadata` and `note_text`. `make_demo_data.py` shows every column the pipeline reads. Use `--site sfdph` for a site without clinical notes or pregnancy data; it runs `cohort.py`, `demographics.py` and `survival.py` only.

To classify symptoms with a real LLM, edit `call_llm` in `classify_symptoms.py` to call the LLM API of your choice, then run `classify_symptoms.py` and `uncoded_comparison.py` without `--fake-llm`. In the paper we used gpt-oss-20b running on a UCSF server so that notes never left UCSF. Check that your data use agreement allows sending notes to an outside API first.

## Citation
