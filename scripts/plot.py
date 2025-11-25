import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(path: Path):
    data = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algo"]
            m = int(row["M"])
            gflops = float(row["gflops"])
            data.setdefault(algo, []).append((m, gflops))
    for algo in data:
        data[algo].sort(key=lambda x: x[0])
    return data


def plot_gemm(data, out_path: Path):
    plt.figure(figsize=(8, 5))
    for algo, pts in data.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=algo)
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("GFLOPS")
    plt.title("GEMM performance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    results_path = Path("results.csv")
    if not results_path.exists():
        print("results.csv not found. Run the benchmark first.")
        return
    data = load_results(results_path)
    out_path = Path("gemm_perf.png")
    plot_gemm(data, out_path)


if __name__ == "__main__":
    main()
