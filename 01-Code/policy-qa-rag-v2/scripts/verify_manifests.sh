#!/usr/bin/env bash
# Verify that the Day 13 copy of RAG v1 (code, tests, corpus) is byte-identical to Day 12.
#
# Checks (all read-only with respect to Day 12):
#   1. Day 13 files match the Day 13 manifests            (nothing changed since the copy)
#   2. Day 12 working files match the same manifests      (copy == Day 12 on disk)
#   3. Day 12 git commit b366fc8 matches the manifests    (copy == committed/pushed v1)
#   4. Reference result files match Day 12's results/     (baseline numbers are the originals)
#
# Usage (from the project root):  bash scripts/verify_manifests.sh
# Exit code 0 only if every check passes.

set -u
cd "$(dirname "$0")/.."
DAY12="../../../Day-12/01-Code/policy-qa-rag"
V1_COMMIT="b366fc8"
fail=0

echo "== Manifest verification ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =="
echo "Day 13 project : $(pwd)"
echo "Day 12 project : $(cd "$DAY12" && pwd)"
echo

for m in manifests/v1-baseline-code.sha256 manifests/corpus.sha256; do
  echo "-- [1] Day 13 files vs $m"
  shasum -a 256 -c --quiet "$m" && echo "   PASS ($(wc -l < "$m" | tr -d ' ') files)" || { echo "   FAIL"; fail=1; }
  echo "-- [2] Day 12 working files vs $m"
  (cd "$DAY12" && shasum -a 256 -c --quiet "$OLDPWD/$m") && echo "   PASS" || { echo "   FAIL"; fail=1; }
  echo "-- [3] Day 12 git commit $V1_COMMIT vs $m"
  mismatches=0
  while read -r hash path; do
    actual=$(git -C "$DAY12" show "$V1_COMMIT:$path" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
    [ "$actual" = "$hash" ] || { echo "   MISMATCH: $path"; mismatches=$((mismatches + 1)); }
  done < "$m"
  [ "$mismatches" -eq 0 ] && echo "   PASS" || { echo "   FAIL ($mismatches files)"; fail=1; }
done

echo "-- [4] Reference results vs Day 12 results/"
for pair in "reference/day12-smoke-results-run1.json:results/smoke-results-run1.json" \
            "reference/day12-smoke-test-log-run1.txt:results/smoke-test-log-run1.txt"; do
  ours="${pair%%:*}"; theirs="$DAY12/${pair#*:}"
  if cmp -s "$ours" "$theirs" && shasum -a 256 -c --quiet manifests/day12-reference-results.sha256; then
    echo "   PASS  $ours"
  else
    echo "   FAIL  $ours"; fail=1
  fi
done

echo
[ "$fail" -eq 0 ] && echo "OVERALL: PASS" || echo "OVERALL: FAIL"
exit "$fail"
