#!/usr/bin/env python3
"""Runs cargo-mutants mutants of arrow-select against the properties and
against arrow-select's own tests, on parallel workers.

`setup` copies an arrow-rs checkout once, then gives each worker its own copy
of it and of this repository, so builds do not share a target directory. A
mutant is applied, arrow-pbt rebuilt, and the properties that exercise the
mutated file run with one seed. Then the same mutant runs
`cargo test -p arrow-select`. Results go to target/mutation/results.jsonl,
so a run never changes the published results.jsonl here.

usage: run.py setup K ARROW_RS | run K MUTANTS.json [LIMIT] [SEED]
"""
import json, os, random, shutil, subprocess, sys, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Under target, which the copies of this repository leave out
WORK = os.path.join(REPO, "target", "mutation")
# The unmutated copy, so the checkout can change during a run
PRISTINE = os.path.join(WORK, "arrow-rs")
RESULTS = os.path.join(WORK, "results.jsonl")

# Properties that exercise each file, directly or through a kernel that
# calls it
PROPS = {
    "take.rs": "take,take_record_batch,law_take_take,law_filter_take,law_interleave,lexsort,coalesce",
    "filter.rs": "filter,law_filter_take,coalesce,nullif",
    "zip.rs": "zip",
    "interleave.rs": "interleave,interleave_record_batch,law_interleave,zip",
    "concat.rs": "concat,law_concat_slices,concat_batches,coalesce",
    "merge.rs": "merge,merge_n",
    "window.rs": "shift",
    "union_extract.rs": "union_extract",
    "coalesce.rs": "coalesce",
    "boolean.rs": "coalesce",
    "byte.rs": "coalesce",
    "list.rs": "coalesce",
    "byte_view.rs": "coalesce",
    "primitive.rs": "coalesce",
    "generic.rs": "coalesce",
    "fixed_size_binary.rs": "coalesce",
    "nullif.rs": "nullif",
    "dictionary.rs": "interleave,law_interleave,concat,coalesce",
}

def worker_dir(k):
    return os.path.join(WORK, f"w{k}")

# Resident memory a worker's command may use, across its process group. A
# mutant can make a loop allocate without end (an iterator that never ends),
# and a few of those exhaust memory. Unmutated, arrow-select's tests peak at
# 3.5 GB and the harness at 1.2 GB
MEMORY_LIMIT_KB = 6 * 1024 * 1024

def group_rss_kb(pgid):
    # Every process's group and resident size, which BSD and Linux `ps` print
    # the same way. Their `-g` options select different things
    out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    rows = (line.split() for line in out.splitlines())
    return sum(int(r[1]) for r in rows if len(r) == 2 and r[0] == str(pgid) and r[1].isdigit())

def sh(cmd, cwd, timeout=None, env=None):
    # One build job and one test thread per worker, so the workers together
    # use about as many threads as there are workers
    e = dict(os.environ, CARGO_BUILD_JOBS="1", RUST_TEST_THREADS="1", **(env or {}))
    p = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e, start_new_session=True)
    out = []
    reader = threading.Thread(target=lambda: out.append(p.stdout.read()))
    reader.start()
    start, verdict = time.time(), None
    while p.poll() is None:
        if timeout and time.time() - start > timeout:
            verdict = "timeout"
        elif group_rss_kb(p.pid) > MEMORY_LIMIT_KB:
            verdict = "memory"
        if verdict:
            os.killpg(p.pid, 9)
            break
        time.sleep(0.5)
    p.wait()
    reader.join()
    return (verdict or p.returncode), "".join(out)

def setup(k):
    w = worker_dir(k)
    os.makedirs(w, exist_ok=True)
    for src, dst in ((PRISTINE, "arrow-rs"), (REPO, "arrow-pbt")):
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "target", "--exclude", ".git", f"{src}/", f"{w}/{dst}/"], check=True)
    # rsync gives the restored files their old modification times, older
    # than a mutant's build left from an earlier run, so Cargo would reuse
    # that build. Touching the mutated crate forces a rebuild
    for root, _, files in os.walk(os.path.join(w, "arrow-rs", "arrow-select")):
        for f in files:
            os.utime(os.path.join(root, f))
    # Patch the worker's arrow-pbt to the worker's arrow-rs
    example = open(os.path.join(REPO, ".cargo", "config.toml.example")).read()
    with open(os.path.join(w, "arrow-pbt", ".cargo", "config.toml"), "w") as f:
        f.write(example.replace("/path/to/arrow-rs", os.path.join(w, "arrow-rs")))

def run_one(k, m, seed):
    w = worker_dir(k)
    path = m["file"]
    src = os.path.join(w, "arrow-rs", path)
    pristine = os.path.join(PRISTINE, path)
    with open(os.path.join(w, "mutant.diff"), "w") as f:
        f.write(m["diff"])
    out = {"name": m["name"], "file": path}
    rc, log = sh(f"patch -s -p0 {path} < ../mutant.diff", os.path.join(w, "arrow-rs"))
    if rc != 0:
        out["result"] = "patch failed"
        shutil.copy(pristine, src)
        return out
    try:
        t = time.time()
        # Built apart from the run, so that a mutant that does not compile
        # counts as unviable rather than killed
        rc, log = sh("cargo build -q", os.path.join(w, "arrow-pbt"))
        out["build_s"] = round(time.time() - t, 1)
        if rc != 0:
            out["harness"] = "unviable"
        else:
            props = PROPS[os.path.basename(path)]
            t = time.time()
            rc, log = sh(f"./target/debug/arrow-pbt run {props} {seed} 1000 0", os.path.join(w, "arrow-pbt"), timeout=400, env={"ARROW_PBT_THREADS": "1"})
            out["harness_s"] = round(time.time() - t, 1)
            # Rows are property, seed, failed, then timings and messages
            rows = [l.split("\t") for l in log.splitlines()]
            failed = [r[0] for r in rows if len(r) > 2 and r[1] == str(seed) and r[2] == "true"]
            out["harness"] = rc if rc in ("timeout", "memory") else ("killed" if failed or rc != 0 else "survived")
            out["harness_failed"] = failed
        t = time.time()
        rc, log = sh("cargo test -q -p arrow-select", os.path.join(w, "arrow-rs"), timeout=900)
        out["tests_s"] = round(time.time() - t, 1)
        out["tests"] = rc if rc in ("timeout", "memory") else ("survived" if rc == 0 else ("unviable" if "error[E" in log else "killed"))
    finally:
        shutil.copy(pristine, src)
    return out

def run(k, mutants_path, limit=None, seed=1):
    ms = json.load(open(mutants_path))
    # A file arrow-rs added since PROPS was written has no properties mapped
    # to it. Its mutants would all survive, which reads as a gap in the
    # properties rather than a missing entry, so stop before running any
    unmapped = sorted({os.path.basename(m["file"]) for m in ms} - PROPS.keys())
    if unmapped:
        sys.exit(f"no properties in PROPS for {', '.join(unmapped)}: add the properties that exercise each file")
    if limit:
        random.Random(0).shuffle(ms)
        ms = ms[:limit]
    # Resume: skip mutants already in results.jsonl
    if os.path.exists(RESULTS):
        done = {json.loads(l)["name"] for l in open(RESULTS)}
        ms = [m for m in ms if m["name"] not in done]
    lock = threading.Lock()
    queue = list(ms)
    results = open(RESULTS, "a")
    def work(i):
        while True:
            with lock:
                if not queue:
                    return
                m = queue.pop(0)
            r = run_one(i, m, seed)
            with lock:
                results.write(json.dumps(r) + "\n")
                results.flush()
                print(json.dumps(r), flush=True)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    if sys.argv[1] == "setup":
        os.makedirs(WORK, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "target", "--exclude", ".git", f"{sys.argv[3]}/", f"{PRISTINE}/"], check=True)
        for k in range(int(sys.argv[2])):
            setup(k)
    else:
        run(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None, int(sys.argv[5]) if len(sys.argv) > 5 else 1)
