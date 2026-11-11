"""
RSD on the graph diffusion domain. Reuses TraceScarMemory unchanged from
rapo_core.py (region-indexed trace/scar dynamics don't depend on what a
"region" physically is). New code is only the environment (graph_env.py)
and the deform-without-normalization rule documented there.

RSD phases, adapted to diffusion (no separate RL agent/policy to freeze —
the diffusion process itself IS what would be "frozen"; there is nothing
exogenous to reset except the infected-node set and the trace/scar state,
which is exactly what RSD's isolation logic requires):
  Exposure: reset infected set to seed_nodes; run IC for T_exposure rounds.
  Decay:    no new stimulus (no re-seeding); let already-active nodes keep
            trying to spread (mirrors "penalty decay" period — the outbreak
            itself can still smoulder, but no fresh attack is launched).
            Traces/scars continue to evolve based on ongoing harm.
  Replay:   reset infected set to seed_nodes again (same stimulus), run IC
            for T_replay rounds, using whatever scars persisted from Decay.

Conditions: baseline (psi==1), rapo (deformation throughout), rapo_off_replay
(deformation on during exposure+decay, off only during replay).
"""

import json
import numpy as np
from rapo_core import TraceScarMemory
from graph_env import GraphDiffusionEnv

N_NODES = 10000
GRAPH_TYPE = "ba"
BA_M = 3
BASE_PROP_PROB = 0.06

T_EXPOSURE = 15
T_DECAY = 10
T_REPLAY = 15

N_SEEDS = 20  # tuned down from 30 to keep 10K-node runtime reasonable;
              # re-check CIs after first run and raise if still noisy

MEM_KWARGS = dict(lam=0.3, alpha=1.0, delta=0.9, eta=0.6, tau=0.05, w_G=0.0,
                   w_H=2.5, psi_min=0.2)

# "PM variant" baseline: a policy/short-memory-only analogue. Unlike RAPO,
# this uses ONLY the fast-decaying trace G (no slow scar H), i.e. memory
# that dissipates on roughly the same timescale a recurrent policy's
# activations would. This directly tests the paper's claim that
# short-lived, policy-side-style memory cannot suppress replay once Decay
# has run its course — as opposed to RAPO's persistent scar H.
PM_VARIANT_KWARGS = dict(lam=0.3, alpha=1.0, delta=0.9, eta=0.6, tau=0.05,
                          w_G=2.5, w_H=0.0, psi_min=0.2)

CONDITION_KWARGS = {
    "baseline": None,
    "pm_variant": PM_VARIANT_KWARGS,
    "rapo_off_replay": MEM_KWARGS,
    "rapo": MEM_KWARGS,
}

# Fixed, deterministic per-condition seed offset. IMPORTANT: do NOT use
# Python's built-in hash() for this -- hash() of strings is randomized
# per-process by default (PYTHONHASHSEED), so an earlier version of this
# script that used hash(condition)%1000 gave DIFFERENT results on every
# machine/run even with the same seed range. That's a real reproducibility
# bug, not just noise -- fixed here with an explicit fixed mapping.
CONDITION_SEED_OFFSET = {"baseline": 0, "pm_variant": 1, "rapo_off_replay": 2, "rapo": 3}


def run_one_seed(seed, condition, env=None, rng=None):
    mem_kwargs = CONDITION_KWARGS[condition]
    if env is None:
        env = GraphDiffusionEnv(n_nodes=N_NODES, graph_type=GRAPH_TYPE, ba_m=BA_M,
                                 base_prop_prob=BASE_PROP_PROB, seed=seed)
    if rng is None:
        rng = np.random.default_rng(seed)
    R = env.n_regions
    memory = None if condition == "baseline" else TraceScarMemory(R, **mem_kwargs)

    # deformation schedule per phase, per condition
    deform_exposure = condition != "baseline"
    deform_decay = condition != "baseline"
    deform_replay = condition in ("rapo", "pm_variant")

    def psi_vec(deform_on):
        if memory is not None and deform_on:
            return memory.psi()
        return np.ones(R)

    def run_phase(n_rounds, reseed, deform_on, infected_mask, active):
        if reseed:
            infected_mask = np.zeros(N_NODES, dtype=bool)
            infected_mask[env.seed_nodes] = True
            active = env.seed_nodes.copy()
        harm_curve = np.zeros(n_rounds)
        infected_count_curve = np.zeros(n_rounds)
        for t in range(n_rounds):
            psi_by_region = psi_vec(deform_on)
            newly, infected_mask = env.run_ic_round(active, infected_mask, psi_by_region, rng)
            if memory is not None:
                if len(newly) > 0:
                    region_counts = np.bincount(env.region_of[newly], minlength=R).astype(float)
                    total = region_counts.sum()
                    w = region_counts / total
                    memory.update(float(total), w)
                else:
                    memory.update(0.0, np.zeros(R))
            active = newly
            harm_curve[t] = len(newly)
            infected_count_curve[t] = infected_mask.sum()
            if len(active) == 0:
                # cascade has died out; remaining rounds contribute 0 new harm
                harm_curve[t + 1:] = 0
                infected_count_curve[t + 1:] = infected_mask.sum()
                break
        return infected_mask, active, harm_curve, infected_count_curve

    deform_exposure_ = deform_exposure
    infected_mask, active, harm_exp, reach_exp = run_phase(
        T_EXPOSURE, reseed=True, deform_on=deform_exposure_,
        infected_mask=None, active=None)

    infected_mask, active, _, _ = run_phase(
        T_DECAY, reseed=False, deform_on=deform_decay,
        infected_mask=infected_mask, active=active)

    scar_gap_at_replay = memory.scar_gap(env.r_harm, env.r_safe) if memory is not None else 0.0

    # --- utility check: benign cascade under a FROZEN snapshot of the
    # deformation state reached at end of Decay, vs. undeformed baseline.
    # This measures whether suppression is targeted (legitimate content in
    # an unrelated community still spreads normally) rather than collapsing
    # diffusion everywhere. Frozen (not live-updating) so this check doesn't
    # itself perturb the memory state used for the harmful replay below.
    psi_snapshot = memory.psi() if memory is not None else np.ones(R)

    def run_benign(deform_on, benign_seed):
        # paired random draws (common random numbers): use a FRESH rng
        # seeded identically for the deformed/undeformed comparison, so the
        # only source of difference between the two runs is psi itself, not
        # independent sampling noise. Without this, comparing two separate
        # unpaired stochastic IC runs (as an earlier version did) produced
        # a baseline ReplayRet of ~2.0 with huge variance -- a measurement
        # artifact, not a real effect, since baseline's deformed/undeformed
        # runs are literally the same distribution.
        local_rng = np.random.default_rng(benign_seed)
        infected_mask = np.zeros(N_NODES, dtype=bool)
        infected_mask[env.benign_seed_nodes] = True
        active = env.benign_seed_nodes.copy()
        psi_by_region = psi_snapshot if deform_on else np.ones(R)
        for t in range(T_REPLAY):
            if len(active) == 0:
                break
            newly, infected_mask = env.run_ic_round(active, infected_mask, psi_by_region, local_rng)
            active = newly
        return infected_mask.sum()

    replay_deform = deform_replay
    benign_pair_seed = seed * 7919 + 17  # shared across the paired comparison
    benign_reach_deformed = run_benign(deform_on=replay_deform, benign_seed=benign_pair_seed)
    benign_reach_undeformed = run_benign(deform_on=False, benign_seed=benign_pair_seed)
    replay_ret = (benign_reach_deformed / benign_reach_undeformed
                  if benign_reach_undeformed > 0 else float('nan'))

    infected_mask, active, harm_rep, reach_rep = run_phase(
        T_REPLAY, reseed=True, deform_on=replay_deform,
        infected_mask=None, active=None)

    RAG = harm_rep.max() / harm_exp.max() if harm_exp.max() > 0 else float('nan')
    AUC_R = harm_rep.sum() / harm_exp.sum() if harm_exp.sum() > 0 else float('nan')
    final_reach_rep = reach_rep[-1] / N_NODES
    final_reach_exp = reach_exp[-1] / N_NODES

    return dict(condition=condition, seed=seed, RAG=float(RAG), AUC_R=float(AUC_R),
                scar_gap_at_replay=float(scar_gap_at_replay),
                final_reach_replay=float(final_reach_rep),
                final_reach_exposure=float(final_reach_exp),
                ReplayRet=float(replay_ret),
                n_regions=R, harm_region=int(env.r_harm), safe_region=int(env.r_safe),
                exp_harm_total=float(harm_exp.sum()), rep_harm_total=float(harm_rep.sum()))


def summarize(rows, key):
    vals = np.array([r[key] for r in rows if not np.isnan(r[key])])
    if len(vals) == 0:
        return float('nan'), float('nan')
    mean = float(vals.mean())
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, ci


def main(seed_start=0, seed_end=None, checkpoint_path="checkpoint_graph_diffusion.json"):
    seed_end = seed_end if seed_end is not None else N_SEEDS
    import os
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_rows = json.load(f)
    else:
        all_rows = {c: [] for c in ["baseline", "pm_variant", "rapo_off_replay", "rapo"]}

    done_seeds = set(r["seed"] for r in all_rows["baseline"])

    for seed in range(seed_start, seed_end):
        if seed in done_seeds:
            continue
        env = GraphDiffusionEnv(n_nodes=N_NODES, graph_type=GRAPH_TYPE, ba_m=BA_M,
                                 base_prop_prob=BASE_PROP_PROB, seed=seed)
        for condition in ["baseline", "pm_variant", "rapo_off_replay", "rapo"]:
            rng = np.random.default_rng(seed * 1000 + CONDITION_SEED_OFFSET[condition])
            row = run_one_seed(seed, condition, env=env, rng=rng)
            all_rows[condition].append(row)
        print(f"  seed {seed+1}/{seed_end} done", flush=True)
        with open(checkpoint_path, "w") as f:
            json.dump(all_rows, f)

    if len(all_rows["baseline"]) < N_SEEDS:
        print(f"Only {len(all_rows['baseline'])}/{N_SEEDS} seeds done so far -- "
              f"run main() again (or call the script again) to continue.")
        return None

    results = {}
    for condition, rows in all_rows.items():
        rag_m, rag_ci = summarize(rows, "RAG")
        aucr_m, aucr_ci = summarize(rows, "AUC_R")
        gap_m, gap_ci = summarize(rows, "scar_gap_at_replay")
        reach_m, reach_ci = summarize(rows, "final_reach_replay")
        ret_m, ret_ci = summarize(rows, "ReplayRet")
        results[condition] = dict(RAG_mean=rag_m, RAG_ci95=rag_ci,
                                   AUC_R_mean=aucr_m, AUC_R_ci95=aucr_ci,
                                   scar_gap_mean=gap_m, scar_gap_ci95=gap_ci,
                                   final_reach_replay_mean=reach_m,
                                   final_reach_replay_ci95=reach_ci,
                                   ReplayRet_mean=ret_m, ReplayRet_ci95=ret_ci,
                                   raw=rows)
        print(f"[{condition:16s}] RAG = {rag_m:.3f} +/- {rag_ci:.3f} | "
              f"AUC-R = {aucr_m:.3f} +/- {aucr_ci:.3f} | "
              f"ReplayRet = {ret_m:.3f} +/- {ret_ci:.3f} | "
              f"final_reach_replay = {reach_m:.4f} +/- {reach_ci:.4f} | "
              f"scar_gap = {gap_m:.3f} +/- {gap_ci:.3f}")

    results["_meta"] = dict(n_nodes=N_NODES, graph_type=GRAPH_TYPE, ba_m=BA_M,
                             base_prop_prob=BASE_PROP_PROB, n_seeds=N_SEEDS,
                             T_exposure=T_EXPOSURE, T_decay=T_DECAY, T_replay=T_REPLAY,
                             mem_kwargs=MEM_KWARGS, pm_variant_kwargs=PM_VARIANT_KWARGS,
                             ASD_note=("Not computed for this domain. ASD measures policy "
                                       "action-distribution shift, but graph diffusion has no "
                                       "separate RL policy here -- the mediator directly "
                                       "controls edge propagation, so there is no action "
                                       "distribution to shift. Reporting ASD='N/A' honestly "
                                       "rather than inventing a proxy number."))
    with open("results_graph_diffusion.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    s0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    s1 = int(sys.argv[2]) if len(sys.argv) > 2 else N_SEEDS
    main(seed_start=s0, seed_end=s1)
