"""Quick partial-fit synthesis for class_count + sample_count from per-run JSONs."""
import json, math, glob
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
RES = BASE / "results"


def load_all(folder):
    return [json.load(open(p)) for p in sorted(glob.glob(str(folder / "*.json")))
            if "summary" not in p]


def fit_gamma(records, group_key):
    """Group by group_key, then per group fit log(stable_rank) ~ γ log(num_params)."""
    by_g = {}
    for r in records:
        by_g.setdefault(r[group_key], []).append(r)
    fits = []
    for g, runs in sorted(by_g.items()):
        by_w = {}
        for r in runs:
            by_w.setdefault(r["width"], []).append(r["stable_rank"])
        widths_sorted = sorted(by_w.keys())
        if len(widths_sorted) < 2:
            continue
        Ns, srs = [], []
        for w in widths_sorted:
            rep = next(r for r in runs if r["width"] == w)
            Ns.append(rep["num_params"])
            srs.append(float(np.mean(by_w[w])))
        Ns, srs = np.array(Ns, float), np.array(srs, float)
        log_n = np.log10(Ns)
        log_sr = np.log10(srs)
        slope, intercept = np.polyfit(log_n, log_sr, 1)
        pred = slope * log_n + intercept
        ss_res = np.sum((log_sr - pred) ** 2)
        ss_tot = np.sum((log_sr - log_sr.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits.append({
            group_key: g, "gamma": float(slope), "r2": float(r2),
            "n_widths": len(widths_sorted), "n_runs": len(runs),
            "params": Ns.tolist(), "srs": srs.tolist(),
        })
    return fits


def main():
    print("=" * 70)
    print("Partial γ fits from local results")
    print("=" * 70)

    cc = load_all(RES / "class_count")
    if cc:
        print(f"\n[CLASS COUNT] {len(cc)} runs")
        fits = fit_gamma(cc, "n_classes")
        print(f"{'n_classes':>10} {'γ':>8} {'R²':>7} {'n_widths':>9} {'n_runs':>7}")
        for f in fits:
            print(f"{f['n_classes']:>10} {f['gamma']:>8.4f} {f['r2']:>7.3f} "
                  f"{f['n_widths']:>9} {f['n_runs']:>7}")
        if len(fits) >= 2:
            xs = [math.log(f["n_classes"]) for f in fits]
            ys = [f["gamma"] for f in fits]
            slope, intercept = np.polyfit(xs, ys, 1)
            pred = np.polyval([slope, intercept], xs)
            r2 = 1 - np.sum((np.array(ys) - pred) ** 2) / np.sum((np.array(ys) - np.mean(ys)) ** 2)
            print(f"\n  γ ≈ {slope:.4f} · log(n_classes) + {intercept:.4f}   (R²={r2:.3f})")

    sc = load_all(RES / "sample_count")
    if sc:
        print(f"\n[SAMPLE COUNT] {len(sc)} runs")
        fits = fit_gamma(sc, "n_samples")
        print(f"{'n_samples':>10} {'γ':>8} {'R²':>7} {'n_widths':>9} {'n_runs':>7}")
        for f in fits:
            print(f"{f['n_samples']:>10} {f['gamma']:>8.4f} {f['r2']:>7.3f} "
                  f"{f['n_widths']:>9} {f['n_runs']:>7}")
        if len(fits) >= 2:
            xs = [math.log(f["n_samples"]) for f in fits]
            ys = [f["gamma"] for f in fits]
            slope, intercept = np.polyfit(xs, ys, 1)
            pred = np.polyval([slope, intercept], xs)
            r2 = 1 - np.sum((np.array(ys) - pred) ** 2) / np.sum((np.array(ys) - np.mean(ys)) ** 2)
            print(f"\n  γ ≈ {slope:.4f} · log(n_samples) + {intercept:.4f}   (R²={r2:.3f})")

    bp_dir = RES / "bandpass_gamma"
    if bp_dir.exists():
        bp_summary_path = bp_dir / "summary.json"
        if bp_summary_path.exists():
            bp = json.load(open(bp_summary_path))
            print(f"\n[β MANIPULATION] {len(bp.get('records',[]))} runs (single seed earlier)")
            for f in bp.get("fits", []):
                print(f"  β={f['beta_actual']:.3f}  γ={f['gamma_variant']:.4f}  R²={f['r2']:.3f}")


if __name__ == "__main__":
    main()
