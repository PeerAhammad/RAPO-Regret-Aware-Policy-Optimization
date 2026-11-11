"""
RSD on the Safety Gym grid-world domain. Same phase structure and
condition set as the graph diffusion experiment; reuses TraceScarMemory
unchanged.

Stimulus (z): goal-seeking is ACTIVE (agent is biased toward the goal,
which sits beyond the hazard wall). Decay: goal-seeking is INACTIVE
(neutral random walk -- no fresh "attempt to cross" is being driven,
mirroring Decay in the other domains).

Deterministic seeding throughout (learned the hard way in graph_diffusion
not to use hash() for this).
"""

import json
import numpy as np
from rapo_core import TraceScarMemory
from safety_gym_env import GridSafetyEnv

GRID_SIZE = 8
T_EXPOSURE = 40
T_DECAY = 20
T_REPLAY = 40
N_SEEDS = 400

MEM_KWARGS = dict(lam=0.25, alpha=1.0, delta=0.9, eta=0.5, tau=0.05, w_G=0.0,
                   w_H=2.0, psi_min=0.25)
PM_VARIANT_KWARGS = dict(lam=0.25, alpha=1.0, delta=0.9, eta=0.5, tau=0.05,
                          w_G=2.0, w_H=0.0, psi_min=0.25)

CONDITION_KWARGS = {"baseline": None, "pm_variant": PM_VARIANT_KWARGS,
                     "rapo_off_replay": MEM_KWARGS, "rapo": MEM_KWARGS}
CONDITION_SEED_OFFSET = {"baseline": 0, "pm_variant": 1, "rapo_off_replay": 2, "rapo": 3}


def run_one_seed(seed, condition):
    mem_kwargs = CONDITION_KWARGS[condition]
    env = GridSafetyEnv(grid_size=GRID_SIZE)
    R = env.n_regions
    memory = None if condition == "baseline" else TraceScarMemory(R, **mem_kwargs)
    rng = np.random.default_rng(seed * 1000 + CONDITION_SEED_OFFSET[condition])

    deform_exposure = condition != "baseline"
    deform_decay = condition != "baseline"
    deform_replay = condition in ("rapo", "pm_variant")

    def psi_vec(deform_on):
        return memory.psi() if (memory is not None and deform_on) else np.ones(R)

    def run_phase(n_steps, goal_active, deform_on, start_cell, track_asd_ref=None):
        cell = start_cell
        harm_curve = np.zeros(n_steps)
        reached_goal = False
        action_dists = []  # for ASD: record (cell, prob_vector_over_neighbors)
        for t in range(n_steps):
            psi_by_region = psi_vec(deform_on)
            nbrs, p = env.base_kernel(cell, goal_active)
            action_dists.append((cell, nbrs, p))
            new_cell, harm = env.step(cell, goal_active, psi_by_region, rng, deform_on)
            if memory is not None:
                w = np.zeros(R)
                w[env._cell_to_idx(new_cell)] = 1.0
                memory.update(harm, w)
            cell = new_cell
            harm_curve[t] = harm
            if cell == env.goal:
                reached_goal = True
        return cell, harm_curve, reached_goal, action_dists

    end_cell, harm_exp, reached_exp, action_dists_exp = run_phase(
        T_EXPOSURE, goal_active=True, deform_on=deform_exposure, start_cell=env.start)

    end_cell, _, _, _ = run_phase(
        T_DECAY, goal_active=False, deform_on=deform_decay, start_cell=end_cell)

    scar_gap_at_replay = memory.scar_gap(env.r_harm, env.r_safe) if memory is not None else 0.0

    # ASD: compare action distributions (policy P0, NOT the deformed
    # kernel) at the same states between Exposure and Replay. Since P0 is a
    # fixed function of (cell, goal_active) only -- never updated, never
    # learned -- this is 0 by construction. We verify rather than assert.
    _, harm_rep, reached_rep, action_dists_rep = run_phase(
        T_REPLAY, goal_active=True, deform_on=deform_replay, start_cell=env.start)

    asd_diffs = []
    for (c1, n1, p1), (c2, n2, p2) in zip(action_dists_exp[:min(len(action_dists_exp), len(action_dists_rep))],
                                            action_dists_rep[:min(len(action_dists_exp), len(action_dists_rep))]):
        if c1 == c2 and n1 == n2:
            asd_diffs.append(0.5 * np.abs(p1 - p2).sum())  # TV distance
    ASD = float(np.mean(asd_diffs)) if asd_diffs else 0.0

    # utility check: does deformation cost progress toward the goal? A
    # binary "did it reach the exact goal cell in T_replay steps" metric
    # was tried first and returned 0% success for EVERY condition
    # including the undeformed baseline (14-cell Manhattan distance, a
    # goal_bias of 0.55 spread across possibly-multiple improving
    # neighbors, and a 40-step budget just aren't enough for reliable
    # exact arrival), which produced ReplayRet = 0/0 = nan for every row.
    # Switched to a continuous progress metric (Manhattan-distance
    # reduction), which is both non-degenerate and arguably a better
    # utility measure anyway -- "did the mediator slow the agent down",
    # not just "did it technically finish in time".
    def replay_progress(deform_on, pair_seed):
        local_rng = np.random.default_rng(pair_seed)
        cell = env.start
        psi_by_region = memory.psi() if (memory is not None and deform_on) else np.ones(R)
        start_dist = abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
        for t in range(T_REPLAY):
            cell, _ = env.step(cell, True, psi_by_region, local_rng, deform_on)
            if cell == env.goal:
                break
        final_dist = abs(cell[0] - env.goal[0]) + abs(cell[1] - env.goal[1])
        return max(0, start_dist - final_dist)  # progress made, >=0

    pair_seed = seed * 7919 + 31
    progress_deformed = replay_progress(deform_replay, pair_seed)
    progress_undeformed = replay_progress(False, pair_seed)

    RAG = harm_rep.max() / harm_exp.max() if harm_exp.max() > 0 else float('nan')
    AUC_R = harm_rep.sum() / harm_exp.sum() if harm_exp.sum() > 0 else float('nan')

    return dict(condition=condition, seed=seed, RAG=float(RAG), AUC_R=float(AUC_R),
                scar_gap_at_replay=float(scar_gap_at_replay), ASD=ASD,
                reached_goal_replay=bool(reached_rep),
                exp_harm_total=float(harm_exp.sum()), rep_harm_total=float(harm_rep.sum()),
                progress_deformed=int(progress_deformed),
                progress_undeformed=int(progress_undeformed))


def summarize(rows, key, is_bool=False):
    vals = np.array([float(r[key]) for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))])
    if len(vals) == 0:
        return float('nan'), float('nan')
    mean = float(vals.mean())
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, ci


def main():
    results = {}
    for condition in ["baseline", "pm_variant", "rapo_off_replay", "rapo"]:
        rows = [run_one_seed(seed, condition) for seed in range(N_SEEDS)]
        rag_m, rag_ci = summarize(rows, "RAG")
        aucr_m, aucr_ci = summarize(rows, "AUC_R")
        asd_m, asd_ci = summarize(rows, "ASD")
        gap_m, gap_ci = summarize(rows, "scar_gap_at_replay")
        total_progress_def = sum(r["progress_deformed"] for r in rows)
        total_progress_undef = sum(r["progress_undeformed"] for r in rows)
        replay_ret = total_progress_def / total_progress_undef if total_progress_undef > 0 else float('nan')
        results[condition] = dict(RAG_mean=rag_m, RAG_ci95=rag_ci,
                                   AUC_R_mean=aucr_m, AUC_R_ci95=aucr_ci,
                                   ASD_mean=asd_m, ASD_ci95=asd_ci,
                                   scar_gap_mean=gap_m, scar_gap_ci95=gap_ci,
                                   ReplayRet=replay_ret,
                                   total_progress_deformed=int(total_progress_def),
                                   total_progress_undeformed=int(total_progress_undef),
                                   raw=rows)
        print(f"[{condition:16s}] RAG = {rag_m:.3f} +/- {rag_ci:.3f} | "
              f"AUC-R = {aucr_m:.3f} +/- {aucr_ci:.3f} | "
              f"ReplayRet = {replay_ret:.3f} | ASD = {asd_m:.4f} +/- {asd_ci:.4f} | "
              f"scar_gap = {gap_m:.3f} +/- {gap_ci:.3f}")

    results["_meta"] = dict(grid_size=GRID_SIZE, n_seeds=N_SEEDS,
                             T_exposure=T_EXPOSURE, T_decay=T_DECAY, T_replay=T_REPLAY,
                             mem_kwargs=MEM_KWARGS, pm_variant_kwargs=PM_VARIANT_KWARGS)
    with open("results_safety_gym.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    main()
