# Cross-lingual representation of CEFR language levels in internal states of multilingual LLMs


This repository contains the code developed as part of my bachelor's thesis **“Cross-lingual representation of CEFR language levels in internal states of multilingual LLMs”**.

- **Author:** Matthias Groppe
- **University / Institute:** University of Münster
- **Advisor:** Prof. Dr. Malte Schilling
- **Submission Date:** 14.10.2026

---
## How to Reproduce the Results

### Data Sources

CEFR-annotated text corpora were used for four languages. With the 
exception of the German learner corpus (obtained directly from the 
authors), all datasets originate from the 
[UniversalCEFR](https://huggingface.co/UniversalCEFR) collection on 
Hugging Face.

| Language | Dataset(s) | Source |
|---|---|---|
| German (DE) | [Ahlers et al.](https://arxiv.org/abs/2512.06483) (Falko + MERLIN + synthetic A1 texts) | obtained directly from authors |
| | ELG_CEFR_DE | [UniversalCEFR/elg_cefr_de](https://huggingface.co/datasets/UniversalCEFR/elg_cefr_de) |
| English (EN) | CEFR-SP | [UniversalCEFR/cefr_sp_en](https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en) |
| | ReadMe_en | [UniversalCEFR/readme_en](https://huggingface.co/datasets/UniversalCEFR/readme_en) |
| Hindi (HI) | ReadMe_hi | [UniversalCEFR/readme_hi](https://huggingface.co/datasets/UniversalCEFR/readme_hi) |
| Arabic (AR) | ReadMe_ar | [UniversalCEFR/readme_ar](https://huggingface.co/datasets/UniversalCEFR/readme_ar) |


### Data Pre-processing

All datasets were converted into a standardised CSV format 
(`title`, `lang`, `source_name`, `format`, `category`, `cefr_level`, 
`license`, `text`) through a five-step pipeline, implemented as 
separate scripts:

1. **`0_counter.py`** — records the CEFR level distribution per dataset 
   prior to processing.
2. **`1_standardise.py`** — converts all datasets to the standardised 
   format.
3. **`2_balance.py`** — downsamples each dataset to its smallest 
   available CEFR level to ensure a balanced distribution.
4. **`3_split.py`** — splits each dataset into training and test data 
   (80/20, fixed seed 42), carried out separately per source dataset 
   *before* merging.
5. **`4_merge.py`** — merges the split datasets per language into 
   a single, unified dataset.

For multilingual training combinations, the Arabic and Hindi training 
portions are additionally oversampled (2.5× and 1.2× respectively) to 
balance their underrepresentation relative to German and English.

### Pipeline: Extraction → Training → Evaluation

To reproduce the experiments, run the scripts in the following order:

1. **Extraction** (`1_extract.py`) — extracts internal states (hidden representations) 
   from the language models for each language/model/layer combination 
   and saves them as `.pt` files.
2. **Training** (`2_train.py`) — trains the probing classifier (MLP) 
   on the extracted internal states.
3. **Evaluation** (`3_evaluate.py`) — evaluates the trained classifier 
   on the held-out test split and computes all reported metrics 
   (accuracy, adjacent accuracy, MAE, confusion matrix, etc.).

4. **Merging** (`cefr_to_excel.py`) — merges the individual evaluation 
   results into a single Excel file for combined analysis.

## Note on Uploaded Results

The result files provided in this repository have been stripped of 
the original text content. Since the licensing terms of some of the 
underlying datasets are not entirely clear, the classified texts were 
removed before upload to avoid any potential licensing conflicts. All 
other fields (predicted/true CEFR levels, confidence scores, metadata, 
etc.) remain intact.

---

## Note on the Use of AI Tools
AI-powered tools were used in the creation of the code:

**Claude** (Anthropic, Sonnet 4.6/5; Opus 4.8/5) - Assistance with structuring, developing, commenting on, and debugging the Python scripts; applies to the entire code.

All code suggestions, results, and recommendations were manually reviewed, executed, and validated against the data before being incorporated. The conceptual design, methodological decisions, and interpretation of the results are entirely my responsibility.