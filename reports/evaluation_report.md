# Northstar RAG Evaluation Report

## Configuration

- Question set: `questions.json`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Retrieval: `FAISS IndexFlatIP with normalized vectors` plus `BM25Okapi`
- `top_k=5`, `min_score=0.450`
- Generation: `deterministic extractive local mode`

## Metric Rubric

- Recall@k: mean fraction of labeled relevant source IDs present in the retrieved source set for supported cases.
- MRR: reciprocal rank of the first relevant retrieved chunk, averaged over supported cases.
- Answer relevance: literal coverage of the expected terms; this is not a semantic judge.
- Citation correctness: expected source IDs are cited and every cited ID was retrieved.
- Faithfulness: exact textual support for extractive answers; live paraphrases are not rated automatically.
- Decision accuracy: supported questions are answered and unsupported questions are abstained.

## Aggregate Results

| Metric | Score |
| --- | ---: |
| Recall@5 | 1.000 |
| MRR | 1.000 |
| Answer relevance | 0.923 |
| Citation correctness | 0.846 |
| Faithfulness, literal | 1.000 |
| Decision accuracy | 0.938 |

## Per-Question Results

| ID | Expected sources | Retrieved sources | Status | Top score | Citations | Relevant | Faithful | Safety events |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| q01 | billing-refunds | billing-refunds, billing-refunds, invoice-corrections, invoice-corrections, programme-extensions | answered | 0.623 | billing-refunds | 1.00 | true | - |
| q02 | billing-refunds | billing-refunds, billing-refunds, billing-refunds, invoice-corrections, programme-extensions | answered | 0.724 | billing-refunds | 1.00 | true | - |
| q03 | invoice-corrections | invoice-corrections, billing-refunds, invoice-corrections, billing-refunds, privacy-data | answered | 0.746 | invoice-corrections | 1.00 | true | - |
| q04 | account-mfa | account-mfa, account-mfa, linux-server-incident, technical-api-limits, linux-server-incident | answered | 0.625 | account-mfa | 1.00 | true | - |
| q05 | account-password | account-password, account-password, account-mfa, account-mfa, billing-refunds | answered | 0.838 | account-password | 1.00 | true | - |
| q06 | programme-extensions | programme-extensions, programme-extensions, privacy-data, programme-attendance, linux-server-incident | answered | 0.505 | programme-extensions | 1.00 | true | - |
| q07 | technical-api-limits | technical-api-limits, technical-api-limits, linux-server-incident, account-mfa, technical-incidents | answered | 0.467 | technical-api-limits | 1.00 | true | - |
| q08 | technical-incidents | technical-incidents, technical-incidents, linux-server-incident, programme-attendance, linux-server-incident | answered | 0.651 | technical-incidents | 1.00 | true | - |
| q09 | privacy-data, technical-incidents | privacy-data, account-mfa, account-mfa, programme-extensions, technical-incidents | abstained | 0.286 | - | 0.00 | n/a | - |
| q10 | retrieved-content-safety | retrieved-content-safety, retrieved-content-safety, invoice-corrections, programme-extensions, privacy-data | answered | 0.574 | retrieved-content-safety | 1.00 | true | filtered_retrieved_instruction:retrieved-content-safety-1 |
| q11 | - | programme-attendance, account-password, technical-incidents, billing-refunds, billing-refunds | abstained | 0.162 | - | n/a | n/a | - |
| q12 | - | invoice-corrections, billing-refunds, account-password, account-mfa, account-mfa | abstained | 0.135 | - | n/a | n/a | - |
| q13 | account-mfa, privacy-data | account-mfa, account-mfa, technical-api-limits, privacy-data, account-password | answered | 0.549 | account-mfa | 1.00 | true | - |
| q14 | programme-extensions | programme-extensions, programme-extensions, invoice-corrections, technical-incidents, billing-refunds | answered | 0.742 | programme-extensions | 1.00 | true | - |
| q15 | programme-attendance | programme-attendance, privacy-data, technical-incidents, billing-refunds, programme-extensions | answered | 0.546 | programme-attendance | 1.00 | true | - |
| q16 | - | linux-server-incident, privacy-data, account-mfa, programme-extensions, invoice-corrections | abstained | 0.101 | - | n/a | n/a | - |

## Failure And Fragility Analysis

- `q09`: status `abstained`, expected sources `privacy-data, technical-incidents`, retrieved `privacy-data, account-mfa, account-mfa, programme-extensions, technical-incidents`. Inspect its evidence and query wording.
- Fragile supported case `q09` has top score `0.286`; add paraphrases before trusting the threshold.
- Fragile supported case `q07` has top score `0.467`; add paraphrases before trusting the threshold.

## Threshold Decision

Configured threshold: `0.450`. On the separate calibration set, it produced `0` false rejections and `0` false acceptances. This is a small tuning fixture, not a reliability guarantee.

| Calibration ID | Expected decision | Top score | Threshold decision |
| --- | --- | ---: | --- |
| cal01 | answer | 0.569 | answer |
| cal02 | answer | 0.503 | answer |
| cal03 | answer | 0.722 | answer |
| cal04 | abstain | 0.266 | abstain |
| cal05 | abstain | 0.441 | abstain |
| cal06 | abstain | 0.291 | abstain |

Similarity alone does not prove policy applicability; inspect near-miss questions and generator decisions too.

## Prioritized Recommendations

1. Improve multi-source coverage and citation completeness for `q09`; retrieve both privacy and incident evidence before answering.
2. Expand calibration with new human-authored near-miss questions; the six supplied cases do not prove that the selected threshold generalizes.
3. Independently review live-generated claims and broaden prompt-injection and privacy tests before operational use.
