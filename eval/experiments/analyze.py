"""Turn the experiments' results into the report's tables and the E2 figure.

Usage, from the repo root, on the folder run_experiments.sh wrote (with e1/,
e2_long/, e2_full_kv/, e3/; missing experiments are skipped):
  python -m eval.experiments.analyze RESULTS_DIR

Writes RESULTS_DIR/analysis/: e1.md, e2.md, e3.md (tables) and, if matplotlib
is installed, e2_gpu_memory.png and .pdf. Reads only JSON and text files, so it
runs without torch or a GPU.

Memory values are in GiB (telemetry's "GB" is 1024**3 bytes). GPU memory is the
peak reserved memory per GPU, largest of the 4 GPUs (a run fails as soon as one
GPU is full). "Copied" is data copied from CPU to GPU for the retrieved chunks;
WorldKV (GPU) keeps its chunks on the GPU, so it has none.
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 16
MIN_CHUNKS = 9  # shortest video for which WorldKV's attention window is complete
A40_GIB = 46068 / 1024  # one A40's memory as reported by the GPU (45.0 GiB)
NAMES = {"gpu_only": "WorldKV (GPU)", "cpu_only": "WorldKV (CPU)", "offline": "Ours",
         "full_kv": "Full KV"}


def seconds(chunks):
    return (12 * chunks - 3) / FPS


def run_names(run_list):
    """Run names of a run list in eval/experiments/ (as eval/runner.py names runs)."""
    with open(os.path.join(HERE, run_list)) as f:
        return [f"{p['scene']}_{p['trajectory']}" for p in json.load(f)["pairs"]]


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def telemetry(out_dir, method, run):
    return read_json(os.path.join(out_dir, "runs", method, run + ".telemetry.json"))


def meta(out_dir, method, run):
    return read_json(os.path.join(out_dir, "runs", method, run + ".json")) or {}


def mean(values):
    values = [v for v in values or [] if v is not None]
    return sum(values) / len(values) if values else None


def run_numbers(t, method):
    """Speed and memory numbers of one run's telemetry (None if missing)."""
    if t is None:
        return None
    copies = method != "gpu_only"
    return {
        "chunks": t.get("num_chunks"),
        "fps": t.get("fps_mean"),
        "chunk_s": mean(t.get("chunk_times_s")),
        "copy_wait_s": mean(t.get("retrieval_s")) if copies else None,
        "copied_gb": t.get("copied_total_gb") if copies else None,
        "gpu_gb": max(t["peak_gpu_mem_reserved_gb"]) if t.get("peak_gpu_mem_reserved_gb") else None,
        "bank_gb": t.get("peak_bank_gb"),
        "rss_gb": t.get("peak_cpu_rss_total_gb"),
    }


def read_checks(out_dir):
    """{run: {"cpu_only": bool, "reference": bool, "bank": bool}} from checks.txt."""
    checks, run = {}, None
    path = os.path.join(out_dir, "checks.txt")
    if not os.path.exists(path):
        return checks
    for line in open(path):
        if line.startswith("--- "):
            run = line[4:].strip()
            checks[run] = {}
        elif run and line.startswith(("PASS", "FAIL")):
            ok = line.startswith("PASS")
            if "equals cpu_only video" in line:
                checks[run]["cpu_only"] = ok
            elif "offline video equals" in line:
                checks[run]["reference"] = ok
            elif "offline bank" in line:
                checks[run]["bank"] = ok
            elif "same retrieved chunks" in line:
                checks[run]["chunks"] = ok
    return checks


def fmt(value, spec=".2f", none="-"):
    if value is None:
        return none
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    return format(value, spec)


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def e1_tables(out_dir):
    """Per-run rows and per-method averages for E1."""
    checks = read_checks(out_dir)
    rows, per_method = [], {m: [] for m in ("gpu_only", "cpu_only", "offline")}
    for run in run_names("e1.json"):
        for method in per_method:
            n = run_numbers(telemetry(out_dir, method, run), method)
            if n is None:
                rows.append([run, NAMES[method], meta(out_dir, method, run).get("status", "missing")]
                            + ["-"] * 6)
                continue
            per_method[method].append(n)
            c = checks.get(run, {})
            identical = (c.get("cpu_only") and c.get("reference")) if method == "offline" else None
            rows.append([run, NAMES[method], fmt(identical, none=""), fmt(n["fps"], ".3f"),
                         fmt(n["chunk_s"]), fmt(n["copy_wait_s"], ".3f"), fmt(n["copied_gb"], ".1f"),
                         fmt(n["gpu_gb"], ".1f"), fmt(n["chunks"], "d")])
    header = ["Run", "Method", "Identical to both baselines", "FPS", "Time per chunk (s)",
              "Copy wait per chunk (s)", "Copied CPU->GPU (GiB)", "Peak GPU per GPU (GiB)", "Chunks"]
    summary = []
    for method, numbers in per_method.items():
        avg = lambda key: mean([n[key] for n in numbers])
        summary.append([NAMES[method], str(len(numbers)), fmt(avg("fps"), ".3f"), fmt(avg("chunk_s")),
                        fmt(avg("copy_wait_s"), ".3f"), fmt(avg("copied_gb"), ".1f"),
                        fmt(max((n["gpu_gb"] for n in numbers if n["gpu_gb"]), default=None), ".1f")])
    summary_header = ["Method", "Videos", "FPS (mean)", "Time per chunk (s)", "Copy wait (s)",
                      "Copied per video (GiB)", "Peak GPU per GPU (GiB, max)"]
    return markdown(summary_header, summary) + "\n\n" + markdown(header, rows)


def chunk_peaks(out_dir, method, run):
    """Per-chunk peak GPU memory (largest GPU), from telemetry or, for a run that
    crashed, from its per-GPU progress files."""
    t = telemetry(out_dir, method, run)
    if t is not None:
        return t.get("chunk_peak_gpu_mem_reserved_gb") or []
    base = os.path.join(out_dir, "runs", method, run + ".telemetry.json")
    per_rank = [read_json(p).get("chunk_peak_gpu_mem_reserved_gb") or []
                for p in sorted(glob.glob(base + ".rank*.progress.json"))]
    if not per_rank:
        return []
    return [max(values) for values in zip(*per_rank)]


def e2_data(long_dir, full_kv_dir):
    """{method: {"points": [(seconds, GiB)], "out_of_memory": seconds or None}}."""
    data = {}
    long_run = run_names("e2_long.json")[0]
    for method in ("gpu_only", "cpu_only", "offline"):
        peaks = chunk_peaks(long_dir, method, long_run)
        m = meta(long_dir, method, long_run)
        oom = None
        if m.get("status") == "failed" and m.get("out_of_memory"):
            oom = seconds(m.get("chunks_done", len(peaks)) + 1)
        data[method] = {"points": [(seconds(k + 1), p) for k, p in enumerate(peaks)
                                   if k + 1 >= MIN_CHUNKS],
                        "out_of_memory": oom}
    points, failed = [], []
    for run in run_names("e2_full_kv.json"):
        chunks = (int(run.rsplit("_", 1)[1]) + 3) // 12
        t = telemetry(full_kv_dir, "full_kv", run)
        if t is not None:
            points.append((seconds(chunks), max(t["peak_gpu_mem_reserved_gb"])))
        elif meta(full_kv_dir, "full_kv", run).get("status") == "failed":
            failed.append(chunks)
    data["full_kv"] = {"points": sorted(points),
                       "out_of_memory": seconds(min(failed)) if failed else None}
    return data


def e2_table(data):
    rows = []
    for method in ("full_kv", "gpu_only", "cpu_only", "offline"):
        d = data[method]
        values = [y for _, y in d["points"]]
        rows.append([NAMES[method], str(len(d["points"])),
                     fmt(min(values), ".1f") if values else "-",
                     fmt(max(values), ".1f") if values else "-",
                     fmt(max((x for x, _ in d["points"]), default=None), ".1f"),
                     fmt(d["out_of_memory"], ".1f", none="none")])
    header = ["Method", "Points", "Lowest (GiB)", "Highest (GiB)", "Longest video that ran (s)",
              "First out of memory (s)"]
    points = ["", "Points (video length s: peak GiB):"]
    for method in ("full_kv", "gpu_only", "cpu_only", "offline"):
        points.append(f"- {NAMES[method]}: " + ", ".join(
            f"{x:.1f}: {y:.1f}" for x, y in data[method]["points"]))
    return markdown(header, rows) + "\n" + "\n".join(points)


def plot_e2(data, path_without_extension):
    """The E2 figure: peak GPU memory per GPU vs video length; stars where a method
    ran out of memory, drawn on the GPU's capacity line. Returns False without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    styles = {"full_kv": ("tab:red", "o"), "gpu_only": ("tab:orange", "s"),
              "cpu_only": ("tab:blue", "^"), "offline": ("tab:green", "D")}
    for method in ("full_kv", "gpu_only", "cpu_only", "offline"):
        color, marker = styles[method]
        d = data[method]
        if d["points"]:
            xs, ys = zip(*d["points"])
            ax.plot(xs, ys, color=color, marker=marker if method == "full_kv" else None,
                    markersize=4, label=NAMES[method])
        if d["out_of_memory"] is not None:
            ax.plot([d["out_of_memory"]], [A40_GIB], marker="*", color=color, markersize=15,
                    markeredgecolor="black", linestyle="none")
    ax.axhline(A40_GIB, color="gray", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], A40_GIB + 0.3, "A40 memory (45 GiB) ", color="gray", fontsize=8,
            ha="right")
    ax.set_xlabel("Video length (s)")
    ax.set_ylabel("Peak GPU memory per GPU (GiB)")
    ax.set_title("GPU memory vs video length (4x A40); * = out of memory")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_without_extension}.{ext}", dpi=150)
    plt.close(fig)
    return True


def e3_table(out_dir):
    checks = read_checks(out_dir)
    rows = []
    for run in run_names("e3.json"):
        cpu = run_numbers(telemetry(out_dir, "cpu_only", run), "cpu_only")
        ours = run_numbers(telemetry(out_dir, "offline", run), "offline")
        path = run.rsplit("_", 2)[-2]
        if cpu is None or ours is None:
            rows.append([path] + ["missing"] + ["-"] * 7)
            continue
        saving = lambda key: (1 - ours[key] / cpu[key]) if cpu[key] and ours[key] else None
        c = checks.get(run, {})
        rows.append([path, fmt(c.get("cpu_only"), none=""),
                     f"{fmt(cpu['bank_gb'], '.1f')} -> {fmt(ours['bank_gb'], '.1f')}",
                     fmt(saving("bank_gb"), ".0%"),
                     f"{fmt(cpu['rss_gb'], '.1f')} -> {fmt(ours['rss_gb'], '.1f')}",
                     fmt(saving("rss_gb"), ".0%"),
                     f"{fmt(cpu['fps'], '.3f')} -> {fmt(ours['fps'], '.3f')}",
                     f"{fmt(cpu['copy_wait_s'], '.3f')} -> {fmt(ours['copy_wait_s'], '.3f')}",
                     f"{fmt(cpu['copied_gb'], '.0f')} -> {fmt(ours['copied_gb'], '.0f')}"])
    header = ["Path", "Identical video", "Bank (GiB)", "Bank saving", "Process memory (GiB)",
              "Process saving", "FPS", "Copy wait per chunk (s)", "Copied CPU->GPU (GiB)"]
    return ("WorldKV (CPU) -> ours.\n\n" + markdown(header, rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", help="Folder with e1/, e2_long/, e2_full_kv/, e3/.")
    args = parser.parse_args()
    out = os.path.join(args.results, "analysis")
    os.makedirs(out, exist_ok=True)
    written = []
    if os.path.isdir(os.path.join(args.results, "e1")):
        with open(os.path.join(out, "e1.md"), "w") as f:
            f.write("# E1: short videos\n\n" + e1_tables(os.path.join(args.results, "e1")) + "\n")
        written.append("e1.md")
    if os.path.isdir(os.path.join(args.results, "e2_long")):
        data = e2_data(os.path.join(args.results, "e2_long"),
                       os.path.join(args.results, "e2_full_kv"))
        with open(os.path.join(out, "e2.md"), "w") as f:
            f.write("# E2: GPU memory vs video length\n\n" + e2_table(data) + "\n")
        written.append("e2.md")
        if plot_e2(data, os.path.join(out, "e2_gpu_memory")):
            written += ["e2_gpu_memory.png", "e2_gpu_memory.pdf"]
        else:
            print("matplotlib not installed: E2 figure skipped")
    if os.path.isdir(os.path.join(args.results, "e3")):
        with open(os.path.join(out, "e3.md"), "w") as f:
            f.write("# E3: long videos\n\n" + e3_table(os.path.join(args.results, "e3")) + "\n")
        written.append("e3.md")
    print(f"wrote {', '.join(written) or 'nothing'} in {out}")


if __name__ == "__main__":
    main()
