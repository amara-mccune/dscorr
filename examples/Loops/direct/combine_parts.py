"""Sum the pieces of direct_vec.py runs made with PART=i/n. Usage: combine_parts.py out.jsonl in1.jsonl [in2.jsonl ...]
Rows without "part" are copied. A value of r is written only when all n pieces are present."""
import json, sys
from collections import defaultdict
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
groups, out = defaultdict(dict), []
for x in rows:
    if not x.get("part"):
        out.append(x)
        continue
    i, n = map(int, x["part"].split("/"))
    key = (x["mu"], x["kappa"], x["shape"], x["r"], x["Pc"], x["nphi"], x["subset"], x["contact"], x["grid"], n)
    groups[key][i] = x
for key, parts in sorted(groups.items(), key=lambda kv: kv[0][3]):
    n = key[-1]
    if len(parts) < n:
        print(f"incomplete: r = {key[3]} ({len(parts)}/{n} pieces)", file=sys.stderr)
        continue
    x = dict(parts[0])
    x["S"] = sum(parts[i]["S"] for i in range(n))
    x["npts"] = sum(parts[i]["npts"] for i in range(n))
    x["cpu"] = round(sum(parts[i]["cpu"] for i in range(n)), 1)
    x["part"] = f"sum of {n}"
    out.append(x)
with open(sys.argv[1], "w") as fh:
    for x in sorted(out, key=lambda x: x["r"]):
        fh.write(json.dumps(x) + "\n")
print(f"{len(out)} values written to {sys.argv[1]}")
