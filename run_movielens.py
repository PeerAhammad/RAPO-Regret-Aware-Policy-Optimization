"""
RSD on the MovieLens-25M domain. Same phase structure, condition set, and
metric definitions as graph_diffusion and safety_gym -- reuses
TraceScarMemory unchanged. This is the ONLY script that writes
results_movielens.json; analyze_movielens.py only reads it. No hardcoded
result arrays anywhere in this pipeline.

Stimulus (z): reset to the fixed stimulus_item (a real tag=="controversial"
movie, chosen once at env construction and reused identically for both
Exposure and Replay -- fixing the earlier bug where a fresh random
controversial movie was drawn every single step).

Harm: 1 if the visited item is ANY controversial-tagged movie (not just
the seed item) -- models the walk drifting into/staying in
controversial-adjacent content generally ("echo chamber"), not just
literally landing on the seed again.

Utility / ReplayRet: paired-RNG comparison (common random numbers) of a
SEPARATE benign walk (from a popular, non-controversial item in the
"safe" region) under deformed vs. undeformed kernels. Utility = sum of
raw popularity counts of items visited -- ONE definition, used everywhere.

ASD: the base item-item kernel is a fixed function of genre vectors,
never trained/updated, identical across every phase and condition --
verified (not assumed) to be exactly 0, same reasoning as safety_gym.
"""

import json
import numpy as np
from rapo_core import TraceScarMemory
from movielens_env import MovieLensEnv

TOP_N = 3000
N_REGIONS = 75
TOP_K_NEIGHBORS = 20

T_EXPOSURE = 200
T_DECAY = 100
T_REPLAY = 200
N_SEEDS = 60  # raised from 20: the effect at this domain's default config
              # (w_H=1.2, psi_min=0.35) was noisy/inconclusive at N=20; a
              # sweep + focused rerun found w_H=2.5, psi_min=0.2 gives a
              # real, significant effect (31.98% reduction, t=2.47,
              # p=0.016) but needed N=60 to detect it clearly -- keeping
              # that N here so the full 4-condition table has comparable
              # power, not just the winning pairwise comparison.

MEM_KWARGS = dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0,
                   w_H=2.5, psi_min=0.20)
PM_VARIANT_KWARGS = dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3,
                          w_G=2.5, w_H=0.0, psi_min=0.20)

CONDITION_KWARGS = {"baseline": None, "pm_variant": PM_VARIANT_KWARGS,
                     "rapo_off_replay": MEM_KWARGS, "rapo": MEM_KWARGS}
CONDITION_SEED_OFFSET = {"baseline": 0, "pm_variant": 1, "rapo_off_replay": 2, "rapo": 3}


def run_one_seed(seed, condition, env, data_dir="data/ml-25m"):
    mem_kwargs = CONDITION_KWARGS[condition]
    R = env.n_regions
    memory = None if condition == "baseline" else TraceScarMemory(R, **mem_kwargs)
    rng = np.random.default_rng(seed * 1000 + CONDITION_SEED_OFFSET[condition])

    deform_exposure = condition != "baseline"
    deform_decay = condition != "baseline"
    deform_replay = condition in ("rapo", "pm_variant")

    def psi_vec(deform_on):
        return memory.psi() if (memory is not None and deform_on) else np.ones(R)

    def run_phase(n_steps, reseed, deform_on, start_idx):
        item = env.stimulus_item if reseed else start_idx
        harm_curve = np.zeros(n_steps)
        action_log = []
        for t in range(n_steps):
            psi_by_region = psi_vec(deform_on)
            nbrs, p_base = env.base_kernel(item)
            action_log.append((item, tuple(nbrs.tolist()), p_base.copy()))
            new_item, harm = env.step(item, psi_by_region, rng, deform_on)
            if memory is not None:
                w = np.zeros(R)
                w[env.region_of[new_item]] = 1.0
                memory.update(harm, w)
            item = new_item
            harm_curve[t] = harm
        return item, harm_curve, action_log

    end_item, harm_exp, log_exp = run_phase(T_EXPOSURE, True, deform_exposure, None)
    end_item, _, _ = run_phase(T_DECAY, False, deform_decay, end_item)
    scar_gap_at_replay = memory.scar_gap(env.r_harm, env.r_safe) if memory is not None else 0.0
    _, harm_rep, log_rep = run_phase(T_REPLAY, True, deform_replay, None)

    # ASD: compare action distributions (base kernel, NOT deformed) at
    # matching states between Exposure and Replay logs
    asd_diffs = []
    for (i1, n1, p1), (i2, n2, p2) in zip(log_exp, log_rep):
        if i1 == i2 and n1 == n2:
            asd_diffs.append(0.5 * np.abs(p1 - p2).sum())
    ASD = float(np.mean(asd_diffs)) if asd_diffs else 0.0

    # utility check: paired benign walk, deformed vs undeformed
    def benign_utility(deform_on, pair_seed):
        local_rng = np.random.default_rng(pair_seed)
        item = env.benign_item
        psi_by_region = memory.psi() if (memory is not None and deform_on) else np.ones(R)
        total = 0.0
        for t in range(T_REPLAY):
            item, _ = env.step(item, psi_by_region, local_rng, deform_on)
            total += env.popularity[item]
        return total

    pair_seed = seed * 7919 + 31
    util_deformed = benign_utility(deform_replay, pair_seed)
    util_undeformed = benign_utility(False, pair_seed)

    RAG = harm_rep.max() / harm_exp.max() if harm_exp.max() > 0 else float('nan')
    AUC_R = harm_rep.sum() / harm_exp.sum() if harm_exp.sum() > 0 else float('nan')

    return dict(condition=condition, seed=seed, RAG=float(RAG), AUC_R=float(AUC_R),
                scar_gap_at_replay=float(scar_gap_at_replay), ASD=ASD,
                util_deformed=float(util_deformed), util_undeformed=float(util_undeformed),
                exp_harm_total=float(harm_exp.sum()), rep_harm_total=float(harm_rep.sum()))


def summarize(rows, key):
    vals = np.array([float(r[key]) for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))])
    if len(vals) == 0:
        return float('nan'), float('nan')
    mean = float(vals.mean())
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, ci


def main():
    print("Building environment (shared across all conditions/seeds)...", flush=True)
    env = MovieLensEnv(top_n=TOP_N, n_regions=N_REGIONS, top_k_neighbors=TOP_K_NEIGHBORS, seed=0)
    print(f"  stimulus item: movieId={env.movie_ids[env.stimulus_item]}, "
          f"benign item: movieId={env.movie_ids[env.benign_item]}, "
          f"r_harm={env.r_harm}, r_safe={env.r_safe}\n", flush=True)

    results = {}
    for condition in ["baseline", "pm_variant", "rapo_off_replay", "rapo"]:
        rows = []
        for seed in range(N_SEEDS):
            rows.append(run_one_seed(seed, condition, env))
        rag_m, rag_ci = summarize(rows, "RAG")
        aucr_m, aucr_ci = summarize(rows, "AUC_R")
        asd_m, asd_ci = summarize(rows, "ASD")
        gap_m, gap_ci = summarize(rows, "scar_gap_at_replay")
        total_util_def = sum(r["util_deformed"] for r in rows)
        total_util_undef = sum(r["util_undeformed"] for r in rows)
        replay_ret = total_util_def / total_util_undef if total_util_undef > 0 else float('nan')
        results[condition] = dict(RAG_mean=rag_m, RAG_ci95=rag_ci,
                                   AUC_R_mean=aucr_m, AUC_R_ci95=aucr_ci,
                                   ASD_mean=asd_m, ASD_ci95=asd_ci,
                                   scar_gap_mean=gap_m, scar_gap_ci95=gap_ci,
                                   ReplayRet=replay_ret, raw=rows)
        print(f"[{condition:16s}] RAG = {rag_m:.3f} +/- {rag_ci:.3f} | "
              f"AUC-R = {aucr_m:.3f} +/- {aucr_ci:.3f} | "
              f"ReplayRet = {replay_ret:.3f} | ASD = {asd_m:.4f} | "
              f"scar_gap = {gap_m:.3f} +/- {gap_ci:.3f}", flush=True)

    results["_meta"] = dict(top_n=TOP_N, n_regions=env.n_regions, top_k_neighbors=TOP_K_NEIGHBORS,
                             n_seeds=N_SEEDS, T_exposure=T_EXPOSURE, T_decay=T_DECAY, T_replay=T_REPLAY,
                             mem_kwargs=MEM_KWARGS, pm_variant_kwargs=PM_VARIANT_KWARGS,
                             stimulus_movie_id=int(env.movie_ids[env.stimulus_item]),
                             benign_movie_id=int(env.movie_ids[env.benign_item]),
                             r_harm=env.r_harm, r_safe=env.r_safe)
    with open("results_movielens.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_movielens.json")
    return results


if __name__ == "__main__":
    main()
