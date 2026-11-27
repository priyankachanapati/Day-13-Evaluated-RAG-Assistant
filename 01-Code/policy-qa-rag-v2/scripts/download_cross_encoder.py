"""One-time download of the Phase 4 cross-encoder, pinned to an exact Hub commit.

    .venv/bin/python scripts/download_cross_encoder.py

* Requested model ID (project plan): ``cross-encoder/ms-marco-MiniLM-L-6-v2``. The Hugging Face
  Hub resolves it to the renamed canonical repository ``cross-encoder/ms-marco-MiniLM-L6-v2``
  (same repository and commit); the canonical ID is used for downloading and loading.
* Pinned to commit ``MODEL_REVISION`` so a later change on the Hub cannot change results.
* Only the files the PyTorch backend needs are downloaded (~92 MB instead of ~890 MB: the repo
  also carries Flax, ONNX and OpenVINO copies).
* Downloaded file sizes are checked against the Hub metadata.
* Writes ``results/cross-encoder-model-info.json`` (revision, files, sizes, config, timings).

Download time is a ONE-TIME cost and is reported separately from per-query latency.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from rag_v2.reranker import ALLOW_PATTERNS, MODEL_ID, MODEL_REVISION, REQUESTED_MODEL_ID  # noqa: E402

OUT = PROJECT_ROOT / "results" / "cross-encoder-model-info.json"


def main() -> int:
    api = HfApi()
    requested = api.model_info(REQUESTED_MODEL_ID)
    info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    print(f"Requested ID : {REQUESTED_MODEL_ID} -> resolves to {requested.id} @ {requested.sha}")
    print(f"Pinned       : {info.id} @ {info.sha}")
    if requested.id != MODEL_ID or info.sha != MODEL_REVISION:
        print("Model ID / revision mismatch - refusing to download.", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=ALLOW_PATTERNS))
    download_s = time.perf_counter() - t0

    expected = {s.rfilename: s.size for s in info.siblings if s.rfilename in ALLOW_PATTERNS}
    files, ok = {}, True
    for name, size in sorted(expected.items()):
        local = path / name
        actual = local.stat().st_size if local.exists() else None
        files[name] = {"hub_size": size, "local_size": actual, "match": actual == size}
        ok &= actual == size
        print(f"  {'OK ' if actual == size else 'BAD'} {name:<26} hub {size:>12,}  local {actual or 0:>12,}")
    config = json.loads((path / "config.json").read_text())
    record = {
        "requested_model_id": REQUESTED_MODEL_ID,
        "canonical_model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "hub_last_modified": str(info.last_modified),
        "license_tags": [t for t in info.tags if t.startswith("license")],
        "pipeline_tag": info.pipeline_tag,
        "files": files,
        "download_bytes": sum(v["local_size"] or 0 for v in files.values()),
        "download_seconds_this_run": round(download_s, 3),
        "note": "download_seconds is ~0 when the files were already in the local Hugging Face cache",
        "config": {
            k: config.get(k)
            for k in (
                "architectures",
                "model_type",
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "max_position_embeddings",
                "id2label",
            )
        },
        "local_snapshot": str(path),
        "huggingface_hub": metadata.version("huggingface-hub"),
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Download time (this run): {download_s:.2f} s for {record['download_bytes']:,} bytes")
    print(f"Config: {record['config']}")
    print(f"Saved {OUT.relative_to(PROJECT_ROOT)} | size check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
