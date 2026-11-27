# Activity 4 – Hit Rate@K and MRR Evaluation (K = 3 and K = 5)

**Requirement.** Use the Day 12 retriever and 10 labelled questions with known relevant chunk IDs; write a compact
Python function for Hit Rate@K and MRR; evaluate at K = 3 and K = 5; compare and interpret.

| Item | Where |
|---|---|
| 10 labelled questions | [`labelled_questions.json`](labelled_questions.json) – IDs, categories, fact labels and known relevant chunk IDs of the first 10 answerable questions of the frozen set [`eval_queries_v1_1.json`](../../evaluation/datasets/eval_queries_v1_1.json) (question texts are read from that file by ID; the held-out guard test forbids copying them) |
| Compact metric function (14 lines incl. docstring) | `hit_rate_and_mrr` in [`hit_rate_mrr.py`](hit_rate_mrr.py) |
| Output | [`results/hit-rate-mrr.json`](results/hit-rate-mrr.json) |
| Unit tests (hand-worked) | [`tests_v2/test_activity4_hit_rate_mrr.py`](../../tests_v2/test_activity4_hit_rate_mrr.py) |

```python
def hit_rate_and_mrr(rankings: list[list[str]], relevant: list[set[str]], k: int) -> tuple[float, float]:
    """Hit Rate@k and MRR@k over queries.

    rankings[i] = retrieved chunk IDs for query i, best first; relevant[i] = its known relevant IDs.
    Hit@k = 1 if any relevant ID is in the top k. RR@k = 1/rank of the first relevant ID in the top k, else 0.
    """
    if not rankings or len(rankings) != len(relevant):
        raise ValueError("need one non-empty relevance set per ranking")
    hits, rr_sum = 0, 0.0
    for ranked, rel in zip(rankings, relevant, strict=True):
        rank = next((i for i, cid in enumerate(ranked[:k], start=1) if cid in rel), None)
        if rank is not None:
            hits, rr_sum = hits + 1, rr_sum + 1.0 / rank
    return hits / len(rankings), rr_sum / len(rankings)
```

```bash
.venv/bin/python activities/activity-4-hit-rate-mrr/hit_rate_mrr.py   # local only, no API calls
```

Rankings come from the actual Day 12 retriever (persisted Day 12 index, 22 chunks). Verified programmatically: the
function equals the project's `evaluation.metrics` at K = 3 and 5, equals the saved Phase 7 v1 records, and the live
top 5 equal the saved Phase 7 rankings for all 10 questions (5/5 checks PASS).

## Measured results

| Metric | K = 3 | K = 5 | Change |
|---|---:|---:|---:|
| Hit Rate | 1.000 | 1.000 | 0.000 |
| MRR | 0.950 | 0.950 | 0.000 |

First relevant rank per question (EV-01 … EV-11, EV-06 excluded as in v1.1): 1, 1, **2** (EV-03), 1, 1, 1, 1, 1, 1, 1.

**Interpretation.** Hit Rate@K tells whether a relevant chunk is retrieved at all within K results; MRR also rewards
putting it near the top (rank 2 counts 0.5). Increasing K can only keep or raise both. Here they are identical at
K = 3 and K = 5 because every first relevant chunk is already at rank 1 or 2: the two extra slots at K = 5 add no
relevant results, only more (non-relevant) context for the LLM. On these 10 questions the Day 12 retriever's
problem is ordering (EV-03 at rank 2), not missing results. A difference between K = 3 and K = 5 would appear only
for questions whose first relevant chunk sits at rank 4 or 5, as in Activity 2 (DEV-01: Hit@3 0, Hit@5 1).
