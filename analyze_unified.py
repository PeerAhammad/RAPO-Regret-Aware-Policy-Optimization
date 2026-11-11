"""Unified cross-domain table on ABSOLUTE replay harm (paired by seed index).
Reads the four results_*.json files (each must contain rep_harm_total per seed)
and, if present, the learned-policy files dl/policy_table_s*_results.json."""
import glob, json, os
import numpy as np
from scipy import stats

CONDS = ["baseline", "pm_variant", "rapo_off_replay", "rapo"]
DOMAINS = [("Graph diffusion", "results_graph_diffusion.json"),
           ("Safety Gym (heuristic policy)", "results_safety_gym.json"),
           ("MovieLens-25M", "results_movielens.json"),
           ("Reddit cascades", "results_reddit.json")]


def change(x, ref):
    return 100.0 * (x.mean() - ref.mean()) / ref.mean()


def pair(x, ref):
    t, p = stats.ttest_rel(x, ref)
    return f"{change(x, ref):+.1f}% (p={p:.2g})"


def load_domain(path):
    res = json.load(open(path))
    res.pop("_meta", None)
    if "rep_harm_total" not in res["baseline"]["raw"][0]:
        return None, res
    harm = {c: np.array([r["rep_harm_total"] for r in res[c]["raw"]], float) for c in CONDS}
    return harm, res


rows = []
for name, path in DOMAINS:
    if not os.path.exists(path):
        print(f"[skip] {name}: {path} not found"); continue
    harm, res = load_domain(path)
    if harm is None:
        print(f"[RERUN NEEDED] {name}: {path} has no rep_harm_total (apply the logging patch and rerun)")
        continue
    ret = res["rapo"].get("ReplayRet", res["rapo"].get("ReplayRet_mean", float("nan")))
    rows.append((name, len(harm["baseline"]), harm["baseline"].mean(), pair(harm["pm_variant"], harm["baseline"]),
                 pair(harm["rapo_off_replay"], harm["baseline"]), pair(harm["rapo"], harm["baseline"]),
                 pair(harm["rapo"], harm["pm_variant"]), ret))

print("\n| Domain | n seeds | Baseline replay harm | PM vs base | RAPO-off@replay vs base | RAPO vs base | RAPO vs PM | RAPO ReplayRet |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.2f} |")

# learned-policy grid-world: one row aggregating several independently trained policies
files = sorted(glob.glob("dl/policy_table_s*_results.json"))
if files:
    per = {k: [] for k in ("rapo_vs_base", "rapo_vs_pm", "pm_vs_base", "off_vs_base", "ret")}
    for f in files:
        d = json.load(open(f))
        h = {c: np.array([r["rep_harm"] for r in d["raw"][c]], float) for c in CONDS}
        per["rapo_vs_base"].append(change(h["rapo"], h["baseline"]))
        per["rapo_vs_pm"].append(change(h["rapo"], h["pm_variant"]))
        per["pm_vs_base"].append(change(h["pm_variant"], h["baseline"]))
        per["off_vs_base"].append(change(h["rapo_off_replay"], h["baseline"]))
        per["ret"].append(d["summary"]["rapo"]["ReplayRet"])
    ms = lambda v: f"{np.mean(v):+.1f}% +/- {np.std(v, ddof=1):.1f} (range {min(v):+.1f} to {max(v):+.1f})" if len(v) > 1 else f"{v[0]:+.1f}%"
    print(f"\nGrid-world, learned MLP policy, {len(files)} independently trained policies (mean +/- SD across policies):")
    print(f"  RAPO vs baseline:           {ms(per['rapo_vs_base'])}")
    print(f"  RAPO vs PM variant:         {ms(per['rapo_vs_pm'])}")
    print(f"  PM variant vs baseline:     {ms(per['pm_vs_base'])}")
    print(f"  RAPO-off@replay vs base:    {ms(per['off_vs_base'])}")
    print(f"  RAPO ReplayRet:             {np.mean(per['ret']):.2f} (range {min(per['ret']):.2f} to {max(per['ret']):.2f})")

print("\nNote: harm units differ by domain (new infections, hazard-cell visits, controversial items, "
      "downvoted comments), so compare percentages across domains, not raw counts.")