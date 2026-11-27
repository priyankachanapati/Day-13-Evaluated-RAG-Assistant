"""Build (or check) the canonical evidence catalog from the frozen corpus.

    .venv/bin/python scripts/build_evidence_catalog.py           # write evaluation/datasets/evidence_catalog.json
    .venv/bin/python scripts/build_evidence_catalog.py --check   # rebuild in memory and compare byte for byte

The catalog has no timestamps, so rebuilding from the same corpus gives an identical file
(and identical SHA-256). It refuses to run if the corpus no longer matches its manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evidence import CATALOG_PATH, build_catalog, catalog_document, dumps  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.ingestion import load_documents  # noqa: E402

CORPUS_MANIFEST = PROJECT_ROOT / "manifests" / "corpus.sha256"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the saved catalog instead of writing it")
    args = parser.parse_args()

    check = subprocess.run(["shasum", "-a", "256", "-c", "--quiet", str(CORPUS_MANIFEST)], cwd=PROJECT_ROOT)
    if check.returncode != 0:
        print("Corpus does not match manifests/corpus.sha256 - refusing to build.", file=sys.stderr)
        return 1
    manifest_sha = hashlib.sha256(CORPUS_MANIFEST.read_bytes()).hexdigest()

    units = build_catalog(load_documents(load_settings().policy_dir))
    content = dumps(catalog_document(units, manifest_sha))
    print(f"Evidence units: {len(units)} | per policy: {dict(Counter(u.policy_id for u in units))}")
    print(f"Corpus manifest SHA-256: {manifest_sha}")
    if args.check:
        same = CATALOG_PATH.read_text(encoding="utf-8") == content
        print(f"Saved catalog identical to rebuild: {'PASS' if same else 'FAIL'}")
        return 0 if same else 1
    if CATALOG_PATH.exists() and CATALOG_PATH.read_text(encoding="utf-8") != content:
        print(f"{CATALOG_PATH} exists with different content - not overwriting.", file=sys.stderr)
        return 1
    CATALOG_PATH.write_text(content, encoding="utf-8")
    print(
        f"Wrote {CATALOG_PATH.relative_to(PROJECT_ROOT)} "
        f"(SHA-256 {hashlib.sha256(content.encode('utf-8')).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
