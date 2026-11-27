# Activity 3 – Precision@5 and Recall@5

**Requirement.** Five test questions with known relevant chunk IDs; get the retriever's top 5; calculate
Precision@5 and Recall@5 per question, showing relevant IDs, retrieved IDs, intersection, numerator and denominator.

    Precision@k = relevant retrieved in top k / total retrieved in top k
    Recall@k    = relevant retrieved in top k / total known relevant

| Item | Where |
|---|---|
| Five labelled questions (frozen dev set) | [`evaluation/datasets/dev_queries.json`](../../evaluation/datasets/dev_queries.json) |
| Known relevant chunk IDs (derived from the frozen fact labels before retrieval) | [`labelled_questions.json`](labelled_questions.json) |
| Executable calculation | [`precision_recall_at_5.py`](precision_recall_at_5.py) (exact fractions) |
| Output | [`results/precision-recall-at-5.json`](results/precision-recall-at-5.json) |
| Unit tests (hand-worked) | [`tests_v2/test_activity3_precision_recall.py`](../../tests_v2/test_activity3_precision_recall.py) |

```bash
.venv/bin/python activities/activity-3-precision-recall-at-5/precision_recall_at_5.py   # local only, no API calls
```

Retriever: the unchanged Day 12 dense retriever over the persisted Day 12 index (22 chunks). A chunk is relevant if it
fully contains one of the question's required evidence sentences. Verified programmatically: the results equal the
project's independent `evaluation.metrics.query_metrics`, and the live top 5 equal the saved Activity B rankings
(10/10 checks PASS).

## Measured results

| Question | Known relevant | Top-5 retrieved (rank order) | Intersection | Precision@5 | Recall@5 |
|---|---|---|---|---:|---:|
| DEV-01 | HR-001-CH02 | HR-001-CH05, HR-001-CH04, HR-002-CH02, **HR-001-CH02**, HR-001-CH03 | HR-001-CH02 | 1/5 = 0.200 | 1/1 = 1.000 |
| DEV-02 | IT-001-CH03 | **IT-001-CH03**, IT-001-CH02, IT-001-CH01, IT-002-CH01, IT-002-CH02 | IT-001-CH03 | 1/5 = 0.200 | 1/1 = 1.000 |
| DEV-03 | IT-001-CH03, IT-002-CH03 | **IT-002-CH03**, **IT-001-CH03**, IT-002-CH01, IT-002-CH02, HR-004-CH01 | both | 2/5 = 0.400 | 2/2 = 1.000 |
| DEV-04 | HR-002-CH03, HR-002-CH04 | **HR-002-CH03**, **HR-002-CH04**, HR-002-CH02, IT-002-CH02, HR-002-CH01 | both | 2/5 = 0.400 | 2/2 = 1.000 |
| DEV-05 | HR-003-CH02, HR-003-CH03 | **HR-003-CH02**, **HR-003-CH03**, HR-003-CH04, HR-002-CH04, HR-002-CH03 | both | 2/5 = 0.400 | 2/2 = 1.000 |
| **Mean** | | | | **0.320** | **1.000** |

**Interpretation.** Recall@5 is perfect: every known relevant chunk is in the top 5. Precision@5 looks low only
because each question has just one or two relevant chunks, so the best possible Precision@5 is 1/5 or 2/5. Every
question reached that maximum. Precision@5 here measures how much non-relevant context a top-5 list carries, not a
retrieval failure.
