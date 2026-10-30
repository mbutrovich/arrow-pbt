#!/usr/bin/env python3
"""Summarizes results.jsonl: per file, what the property harness and
arrow-select's own tests kill, and the mutants only one of them kills.

usage: summarize.py [results.jsonl]
"""
import collections, json, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results.jsonl")
rs = [json.loads(l) for l in open(path)]

def caught(x):
    return x in ("killed", "timeout", "memory")

viable = [r for r in rs if r.get("tests") != "unviable" and r.get("harness") != "unviable" and r.get("result") != "patch failed"]
print(f"{len(rs)} mutants, {len(viable)} viable\n")
print(f"{'file':28s} {'viable':>6s} {'harness':>8s} {'tests':>6s} {'both':>5s} {'only h':>6s} {'only t':>6s} {'neither':>7s}")
by_file = collections.defaultdict(list)
for r in viable:
    by_file[r["file"].split("/")[-1]].append(r)
totals = collections.Counter()
for f, xs in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
    c = collections.Counter()
    for r in xs:
        h, t = caught(r.get("harness")), caught(r.get("tests"))
        c["viable"] += 1
        c["harness"] += h
        c["tests"] += t
        c["both"] += h and t
        c["only h"] += h and not t
        c["only t"] += t and not h
        c["neither"] += not h and not t
    totals.update(c)
    print(f"{f:28s} {c['viable']:6d} {c['harness']:8d} {c['tests']:6d} {c['both']:5d} {c['only h']:6d} {c['only t']:6d} {c['neither']:7d}")
print(f"{'total':28s} {totals['viable']:6d} {totals['harness']:8d} {totals['tests']:6d} {totals['both']:5d} {totals['only h']:6d} {totals['only t']:6d} {totals['neither']:7d}")

for label, pick in [("Killed only by the harness", lambda r: caught(r.get("harness")) and not caught(r.get("tests"))),
                    ("Killed only by arrow-select's tests", lambda r: caught(r.get("tests")) and not caught(r.get("harness")))]:
    xs = [r for r in viable if pick(r)]
    print(f"\n{label}: {len(xs)}")
    for r in xs:
        print(f"  {r['name']}")
