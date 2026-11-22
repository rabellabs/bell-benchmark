#!/usr/bin/env python3
"""A rerun on a fresh machine against the first: two TABLE.json, number by
number. A number is the same when the fresh machine's point lies inside the first machine's 95% interval, or within a tolerance of
it (ms a turn: 3 ms, twice the ±1.5 ms a pass seen between passes; LLM calls a turn: 0.005). The end to end numbers carry new live
measurements, so only the interval test applies to them. Prints each number and SAME or DIFFERENT; exit 0 only when all are the same.

  python bench/compare.py --a FETCH_1/out/TABLE.json --b FETCH_2/out/TABLE.json [--json OUT.json]
"""
import argparse
import json
import pathlib
import sys

TOL = {"ms": 3.0, "calls": 0.005}


def point_iv(x):
    if "mean" in x and "interval" in x:
        return x["mean"], x["interval"]
    if "q50" in x:
        return x["q50"], x["q50_interval"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    A = json.load(open(a.a, encoding="utf-8"))
    B = json.load(open(a.b, encoding="utf-8"))
    rows, same_all = [], True
    for k in sorted(set(A["numbers"]) | set(B["numbers"])):
        x, y = A["numbers"].get(k), B["numbers"].get(k)
        if x is None or y is None:
            rows.append({"number": k, "same": False, "why": "in one table only"})
            same_all = False
            continue
        pa, ia = point_iv(x)
        pb, _ = point_iv(y)
        if pa is None or pb is None:
            continue
        inside = ia[0] <= pb <= ia[1]
        kind = k.rsplit(".", 1)[-1]
        tol = TOL.get(kind) if not k.endswith("e2e") else None
        near = tol is not None and abs(pb - pa) <= tol
        same = inside or near
        same_all &= same
        rows.append({"number": k, "first": pa, "first_interval": ia, "fresh": pb, "diff": pb - pa, "inside": inside, "within_tol": near,
                     "same": same})
    for r in rows:
        if "first" in r:
            print(f"{'SAME     ' if r['same'] else 'DIFFERENT'} {r['number']}: {r['first']:.4f} [{r['first_interval'][0]:.4f}, "
                  f"{r['first_interval'][1]:.4f}] -> {r['fresh']:.4f} ({r['diff']:+.4f})")
        else:
            print(f"DIFFERENT {r['number']}: {r['why']}")
    if A.get("mock") or B.get("mock"):
        print("(a dry run: the end to end numbers are the stand-in's)")
    print("The two tables are the same." if same_all else "The two tables differ.")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"same": same_all, "rows": rows}, indent=1), encoding="utf-8")
    sys.exit(0 if same_all else 1)


if __name__ == "__main__":
    main()
