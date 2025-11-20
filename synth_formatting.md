# Synthetic Data Output Schema (Required for Evaluation Pipeline)

This document specifies the **exact output format** required for all synthetic data generation models (GReaT, diffusion baselines, random-walk method, etc.).  
All teams **must** follow this schema so the shared evaluation pipeline (Wasserstein distance, classifier p-test, multi-hot encoding) can run correctly.

If your synthetic data does not follow this format, **the evaluation will fail**.

---

## 1. Required Output File

Each method must output a **single CSV file** containing synthetic visit-fact rows.

**Example filename:**


---

## 2. Required Table Structure

Each row in the CSV must represent **one feature occurring in one visit**.

This is a **long format** table (not wide).

### Required Columns (names must match exactly)

| Column name     | Type        | Description |
|-----------------|-------------|-------------|
| `visit_id`      | string/int  | Unique ID for each synthetic visit. |
| `person_id`     | string/int  | Patient ID for the visit (can be reused or sampled from real data). |
| `provider_id`   | string/int  | Provider associated with the feature. Must come from the real provider ID space. |
| `feature_type`  | string      | Category of the feature (must match real categories exactly, e.g., `"DRUG"`, `"COND"`). |
| `concept_id`    | string/int  | Code/identifier of the feature. Must be drawn from real data concept IDs. |

### Extra Columns  
You may include additional columns (e.g., `value`, `dose`, `timestamp`).  
**These will be ignored by the evaluator.**

---

## 3. Semantics of Each Row

Each row must encode:

> **(visit_id, feature)** where the feature is defined by `(feature_type, concept_id)`.

Example:


---

## 4. Allowed Values for `feature_type`

Must match real data exactly. Common values include:

- `DRUG` — medication
- `COND` — condition/diagnosis
- `PROC` — procedure
- `SPEC` — provider specialty (optional)
- `DEMO_*` — demographic features (optional)
- `LAB` — lab measurements (optional)

Do **not** invent new category labels.

---

## 5. Allowed Values for `concept_id`

`concept_id` values must be drawn from the **real dataset’s concept vocabulary**.

- No new IDs.
- No random IDs.
- No new invented concepts.

This is required to ensure vocabulary alignment during evaluation.

---

## 6. How the Evaluation Pipeline Uses This Data

Your synthetic CSV is consumed by the evaluation script to:

### 1. Build a shared vocabulary  
Each unique `(feature_type, concept_id)` pair becomes a feature token.

### 2. Build visit × feature multi-hot matrices  
Rows represent visits.  
Columns represent tokens.

### 3. Compute Wasserstein distances  
The evaluator counts per-visit:

- `#drugs` = number of unique entries where `feature_type == "DRUG"`
- `#conditions` = number where `feature_type == "COND"`

### 4. Train a classifier for the p-test  
The model distinguishes real vs synthetic vectors to compute:

- AUC  
- Accuracy  
- Approximate p-value

If the CSV follows the schema above, evaluation runs fully automatically.

---

## 7. Common Mistakes (Do Not Do)

❌ Do **not** output a wide-format table  
→ Must use long format (one row per (visit, feature)).

❌ Do **not** create new feature_type labels  
→ Must match the real dataset.

❌ Do **not** invent new concept IDs  
→ Only use real concept IDs.

❌ Do **not** combine multiple features into one row  
→ Each (visit, feature) pair must be separate.

❌ Do **not** omit required columns  
→ All five required columns must be present.

---

## 8. Minimal Valid Example


This fully satisfies the schema.

---

## 9. Optional Enhancements

You may add any number of optional attributes:

- timestamps (`start_date`, `end_date`)
- dosage information
- lab values
- demographics
- sequence/order indicators
- synthetic provider sampling strategies

These **do not affect evaluation** and will be ignored.

---

## 10. Summary (What Your Model Must Output)

Each model must produce a **synthetic long-format visit-fact CSV** where:

- Each row is **one feature** from **one visit**
- Required columns:
- `feature_type` and `concept_id` must match the real dataset’s vocabulary

If you follow this schema, your synthetic data will be directly compatible with the evaluation pipeline.

---
