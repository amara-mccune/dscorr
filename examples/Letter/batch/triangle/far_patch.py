"""far_patch.py <out.json>: far patch of a brute-force record, in place; no-op if B_150000 is
present."""
import json
import sys

from bf_run import far_patch

if __name__ == "__main__":
    fn = sys.argv[1]
    rec = json.load(open(fn))
    if "B_150000" not in rec:
        rec = far_patch(rec)
        json.dump(rec, open(fn, "w"))
