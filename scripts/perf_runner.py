import argparse
import csv
import json
import subprocess
import shutil
import os
from pathlib import Path
from statistics import mean


def run_binary(exe: Path):
    print(f"[perf_runner] Running {exe}")
    res = subprocess.run([str(exe)], capture_output=False)
    if res.returncode != 0:
        raise RuntimeError(f"Executable failed with code {res.returncode}")


def summarize(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "algo": r["algo"],
                    "M": int(r["M"]),
                    "N": int(r["N"]),
                    "K": int(r["K"]),
                    "ms": float(r["ms"]),
                    "gflops": float(r["gflops"]),
                    "valid": r.get("valid", "yes")
                })
            except Exception:
                continue
    if not rows:
        raise RuntimeError("No data found in results.csv")

    summary = {}
    for r in rows:
        key = (r["algo"], r["M"], r["N"], r["K"])
        summary.setdefault(key, []).append(r)

    out = []
    for (algo, M, N, K), lst in summary.items():
        ms_vals = [x["ms"] for x in lst]
        g_vals = [x["gflops"] for x in lst]
        valid_rate = sum(1 for x in lst if str(x["valid"]).lower() == "yes") / len(lst)
        out.append({
            "algo": algo,
            "M": M,
            "N": N,
            "K": K,
            "runs": len(lst),
            "ms_mean": mean(ms_vals),
            "ms_min": min(ms_vals),
            "gflops_mean": mean(g_vals),
            "gflops_max": max(g_vals),
            "valid_rate": valid_rate,
        })
    return out


def write_outputs(data, out_csv: Path, out_json: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["algo", "M", "N", "K", "runs", "ms_mean", "ms_min", "gflops_mean", "gflops_max", "valid_rate"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    with out_json.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[perf_runner] Wrote {out_csv} and {out_json}")


def find_ncu():
    # 优先环境变量
    env_path = os.environ.get("NCU_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return str(p)
    # 常见安装路径
    candidates = [
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.1\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\ncu.exe"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\ncu.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    # fallback to PATH
    return shutil.which("ncu") or shutil.which("ncu.exe")


def run_ncu(exe: Path, kernels):
    ncu_cmd = find_ncu()
    if not ncu_cmd:
        print("[perf_runner] ncu 未找到，请将 ncu 加入 PATH 或设置环境变量 NCU_PATH 指向 ncu.exe，跳过 profiling。")
        return []

    reports = []
    for k in kernels:
        cmd = [ncu_cmd, "--set", "full", "--kernel-name", k, str(exe)]
        print(f"[perf_runner] Profiling {k} with ncu...")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"[perf_runner] 无法执行 ncu 命令，请检查路径或权限，跳过 {k}")
            continue
        if res.returncode != 0:
            print(f"[perf_runner] ncu failed for {k}: {res.stderr}")
            continue
        report_path = Path("reports")
        report_path.mkdir(exist_ok=True)
        safe_name = k.replace(" ", "_").replace(":", "_")
        out_file = report_path / f"ncu_{safe_name}.txt"
        out_file.write_text(res.stdout)
        reports.append({"kernel": k, "file": out_file.as_posix()})
        print(f"[perf_runner] Saved ncu report to {out_file}")
    # write index
    if reports:
        idx_path = Path("reports/ncu_index.json")
        idx_path.parent.mkdir(exist_ok=True)
        idx_path.write_text(json.dumps(reports, indent=2))
        print(f"[perf_runner] Wrote ncu index to {idx_path}")
    return reports


def main():
    parser = argparse.ArgumentParser(description="Run matmul benchmarks and summarize results.")
    parser.add_argument("--exe", type=Path, default=Path("build/bin/matmul_test.exe"), help="Path to executable")
    parser.add_argument("--csv", type=Path, default=Path("results.csv"), help="Path to results.csv")
    parser.add_argument("--out-csv", type=Path, default=Path("scripts/perf_summary.csv"), help="Output summary CSV")
    parser.add_argument("--out-json", type=Path, default=Path("scripts/perf_summary.json"), help="Output summary JSON")
    parser.add_argument("--skip-run", action="store_true", help="Skip running executable, only summarize")
    parser.add_argument("--ncu", nargs="*", default=[], help="Kernel names to profile with ncu")
    args = parser.parse_args()

    if not args.skip_run:
        run_binary(args.exe)
    data = summarize(args.csv)
    write_outputs(data, args.out_csv, args.out_json)

    if args.ncu:
        run_ncu(args.exe, args.ncu)


if __name__ == "__main__":
    main()
