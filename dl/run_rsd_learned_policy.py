"""RSD on the grid-world with a LEARNED, FROZEN policy. Pure numpy (policy is a table)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from rapo_core import TraceScarMemory
from dl.policy_env import LearnedPolicyEnv

T_EXP, T_DECAY, T_REP, N_SEEDS = 60, 20, 60, 400
MEM = dict(lam=0.1, alpha=1.0, delta=0.97, eta=0.5, tau=0.05, w_G=0.0, w_H=4.0, psi_min=0.1)
PM = dict(lam=0.25, alpha=1.0, delta=0.9, eta=0.5, tau=0.05, w_G=2.0, w_H=0.0, psi_min=0.25)
COND = {"baseline": None, "pm_variant": PM, "rapo_off_replay": MEM, "rapo": MEM}


def run_phase(env, memory, n, goal_active, deform_on, cell, rng, R, learn=True):
    harm = np.zeros(n); goals = 0
    for t in range(n):
        psi = memory.psi() if (memory is not None and deform_on) else np.ones(R)
        cell, h = env.step(cell, goal_active, psi, rng, deform_on)
        if memory is not None and learn:
            w = np.zeros(R); w[env._cell_to_idx(cell)] = 1.0
            memory.update(h, w)
        harm[t] = h
        if goal_active and cell == env.goal:
            goals += 1; cell = env.start          # respawn -> repeated crossing attempts
    return cell, harm, goals


def run_one(table, seed, cond):
    env = LearnedPolicyEnv(table); R = env.n_regions
    mem = None if cond == "baseline" else TraceScarMemory(R, **COND[cond])
    rng = np.random.default_rng(seed)                 # common random numbers across conditions
    d_exp = cond != "baseline"; d_rep = cond in ("rapo", "pm_variant")
    cell, h_exp, _ = run_phase(env, mem, T_EXP, True, d_exp, env.start, rng, R)
    cell, _, _ = run_phase(env, mem, T_DECAY, False, d_exp, cell, rng, R)
    gap = mem.scar_gap(env.r_harm, env.r_safe) if mem is not None else 0.0
    _, h_rep, _ = run_phase(env, mem, T_REP, True, d_rep, env.start, rng, R)

    def probe(on):  # utility probe: frozen psi, paired RNG, deformed vs undeformed
        r2 = np.random.default_rng(seed * 7919 + 31)
        _, _, g = run_phase(env, mem, T_REP, True, on, env.start, r2, R, learn=False)
        return g
    g_def, g_undef = probe(d_rep), probe(False)
    return dict(seed=seed, exp_harm=float(h_exp.sum()), rep_harm=float(h_rep.sum()),
                scar_gap=float(gap), goals_def=int(g_def), goals_undef=int(g_undef))


def main(table_path):
    from scipy import stats
    table = np.load(table_path)
    out = {cond: [run_one(table, s, cond) for s in range(N_SEEDS)] for cond in COND}
    summ = {}
    for cond, rows in out.items():
        rep = np.array([r["rep_harm"] for r in rows]); exp = np.array([r["exp_harm"] for r in rows])
        ok = exp > 0
        aucr = rep[ok].sum() / exp[ok].sum()
        ret = sum(r["goals_def"] for r in rows) / max(1, sum(r["goals_undef"] for r in rows))
        ci = 1.96 * rep.std(ddof=1) / np.sqrt(len(rep))
        summ[cond] = dict(rep_harm_mean=rep.mean(), rep_harm_ci=ci,
                          exp_harm_mean=exp.mean(), AUC_R_pooled=aucr, ReplayRet=ret)
        print(f"{cond:16s} abs replay harm {rep.mean():6.2f} +/- {ci:.2f} | "
              f"exposure harm {exp.mean():6.2f} | AUC-R(pooled) {aucr:5.2f} | ReplayRet {ret:.2f}")
    base = np.array([r["rep_harm"] for r in out["baseline"]])
    print("\nPaired t-tests on ABSOLUTE replay harm (same seeds):")
    for c in ("rapo", "pm_variant", "rapo_off_replay"):
        x = np.array([r["rep_harm"] for r in out[c]])
        t, p = stats.ttest_rel(x, base)
        print(f"  {c:16s} vs baseline: change {100*(x.mean()-base.mean())/base.mean():+6.1f}%  t={t:6.2f}  p={p:.2g}")
    x, y = (np.array([r["rep_harm"] for r in out[k]]) for k in ("rapo", "pm_variant"))
    t, p = stats.ttest_rel(x, y)
    print(f"  rapo vs pm_variant: change {100*(x.mean()-y.mean())/y.mean():+6.1f}%  t={t:6.2f}  p={p:.2g}")
    json.dump(dict(summary=summ, raw=out), open(table_path.replace(".npy", "_results.json"), "w"))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dl/policy_table.npy")