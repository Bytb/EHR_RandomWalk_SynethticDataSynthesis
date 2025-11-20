# Synthetic Data Output Schema (Required Format for Evaluation)

This document defines the exact output format that all baseline models must follow when generating synthetic visit-level data. The evaluation pipeline expects a **processed per-visit CSV** with a specific camelCase schema. Your synthetic output must match this schema exactly.

---

## 1. Output File

Each model must produce a single CSV file containing **one row per synthetic visit**.

**Example filename:**
synth_visits_processed.csv


---

## 2. Required Columns (Names Must Match Exactly)

Your CSV must contain the following columns:

| Column Name             | Description |
|-------------------------|-------------|
| `visitID`               | Unique identifier for each synthetic visit. Must not be null. |
| `personID`              | Patient identifier for the visit. May be null. |
| `providerID`            | Provider identifier for the visit. May be null. |
| `drug_concept_ids`      | A list of drug concept IDs associated with the visit. May be empty or null. |
| `condition_concept_ids` | A list of condition concept IDs associated with the visit. May be empty or null. |

These five columns must appear exactly as written.

---

## 3. Column Rules

### `visitID`
- Must be present and non-null.
- Should uniquely represent a synthetic visit.
- Evaluation assumes one row per visit.

### `personID`
- Allowed to be null.
- Null indicates the model did not assign a patient.

### `providerID`
- Allowed to be null.
- Null indicates the model did not assign a provider.

### `drug_concept_ids` and `condition_concept_ids`
- Represent lists of concept IDs in string form.
- Allowed to be empty or null.
- The following separators are all acceptable:

"111 222 333"
"111;222;333"
"111,222,333"
"111|222|333"


- These fields may also contain `""`, `" "`, or `null` for visits with no drugs or conditions.

---

## 4. Required Structure Per Row

Each row of your CSV must represent:

> A single synthetic visit and all drug/condition concepts associated with that visit.

This means:
- One row = one visit.
- Concept lists belong entirely inside that row.
- Do not create one row per concept.

---

## 5. Valid Examples

The following rows demonstrate acceptable formatting:

visitID,personID,providerID,drug_concept_ids,condition_concept_ids
10001,501,2001,"111;222","9001 9002"
10002,,2007,"","9001"
10003,889,,"555|777|888",""
10004,,,,""


All of the above are valid:
- Null `personID` or `providerID` is allowed.
- Empty or null concept lists are allowed.
- Lists may use any common delimiter.

---

## 6. Format Requirements Summary

Your synthetic CSV must:

- Use **camelCase** column names exactly as shown.
- Provide **one row per visit**.
- Allow null `personID` and `providerID`.
- Allow null or empty concept ID lists.
- Store concept lists as a **string**, using any common delimiter.

Your synthetic CSV must not:

- Change column names.
- Use snake_case or different naming.
- Spread a visit across multiple rows.
- Use nested JSON structures.
- Omit required columns.

---

## 7. Validation

You can test your file using the validator: method/validate_synth.py

Edit the config block, then run:

```python
SYNTH_PATH = "../data/synth/synth_visits_processed.csv"

