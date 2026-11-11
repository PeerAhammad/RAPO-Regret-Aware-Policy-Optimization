"""RAPO / RSD demo site: live replay-suppression experiment + saved results for four domains."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from rapo_core import TraceScarMemory
from safety_gym_env import GridSafetyEnv
from dl.policy_env import LearnedPolicyEnv, ACTIONS
from dl.run_rsd_learned_policy import run_phase

T_EXP, T_DECAY, T_REP = 60, 20, 60
CONDS = ["baseline", "pm_variant", "rapo_off_replay", "rapo"]
LABEL = {"baseline": "Baseline (no memory)", "pm_variant": "Fast trace only",
         "rapo_off_replay": "RAPO-off@Replay (ablation)", "rapo": "RAPO"}
SHORT = {"baseline": "Baseline", "pm_variant": "Fast trace\nonly", "rapo_off_replay": "RAPO-off\n@Replay", "rapo": "RAPO"}

# palette (matches style.css)
CREAM, PAPER, LINE = "#f5f0e1", "#fbf8ee", "#dcd6c2"
SAGE, SKY, SAND, GREY = "#4d6b57", "#2f6d95", "#b9975b", "#8c8a80"
INK, MUTED = "#26302b", "#5b665f"
PHASE_TINT = ("#d9eaf5", "#f0e4c8", "#e3ecdb")   # exposure, decay, replay
COLOR = {"baseline": GREY, "pm_variant": SKY, "rapo_off_replay": SAND, "rapo": SAGE}

PM_KW = dict(lam=0.25, alpha=1.0, delta=0.9, eta=0.5, tau=0.05, w_G=2.0, w_H=0.0, psi_min=0.25)
DOMAINS = [("Graph diffusion", "results_graph_diffusion.json"),
           ("Safety Gym (heuristic policy)", "results_safety_gym.json"),
           ("MovieLens-25M", "results_movielens.json"),
           ("Reddit cascades", "results_reddit.json")]
REPO_URL = "https://github.com/PeerAhammad/Regret-Aware-Policy-Optimization"

st.set_page_config(page_title="RAPO: stopping harm from coming back", layout="wide")
try:
    st.markdown(f"<style>{open(os.path.join(ROOT, 'style.css'), encoding='utf-8').read()}</style>",
                unsafe_allow_html=True)
except OSError:
    st.warning("style.css not found, showing the default theme.")

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": PAPER, "savefig.facecolor": CREAM,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
})


# ---------- policies ----------
def heuristic_table():
    e = GridSafetyEnv(grid_size=8); N = 8
    T = np.zeros((N * N, 5))
    for r in range(N):
        for c in range(N):
            nbrs, p = e.base_kernel((r, c), True)
            for (rr, cc), pr in zip(nbrs, p):
                T[r * N + c, ACTIONS.index((rr - r, cc - c))] = pr
    return T


def policy_options():
    opts = {"Hand-written goal-seeker": None}
    for f in sorted(glob.glob(os.path.join(ROOT, "dl", "policy_table_s*.npy"))):
        if "standin" in f:
            continue
        n = os.path.basename(f).replace("policy_table_s", "").replace(".npy", "")
        opts[f"Trained neural policy #{n}"] = f
    return opts


# ---------- live experiment ----------
def simulate(env, cond, seed, mem_kw):
    R = env.n_regions
    kw = {"baseline": None, "pm_variant": PM_KW, "rapo_off_replay": mem_kw, "rapo": mem_kw}[cond]
    mem = None if kw is None else TraceScarMemory(R, **kw)
    rng = np.random.default_rng(seed)
    d_exp = cond != "baseline"; d_rep = cond in ("rapo", "pm_variant")
    cell, h_exp, _ = run_phase(env, mem, T_EXP, True, d_exp, env.start, rng, R)
    cell, h_dec, _ = run_phase(env, mem, T_DECAY, False, d_exp, cell, rng, R)
    H = mem.H.copy() if mem is not None else np.zeros(R)
    _, h_rep, _ = run_phase(env, mem, T_REP, True, d_rep, env.start, rng, R)

    def probe(on):
        r2 = np.random.default_rng(seed * 7919 + 31)
        return run_phase(env, mem, T_REP, True, on, env.start, r2, R, learn=False)[2]
    return np.concatenate([h_exp, h_dec, h_rep]), H, probe(d_rep), probe(False)


@st.cache_data(show_spinner=False)
def run_live(policy_path, w_H, psi_min, lam, delta, n_seeds):
    table = heuristic_table() if policy_path is None else np.load(policy_path)
    env = LearnedPolicyEnv(table)
    mem_kw = dict(lam=lam, alpha=1.0, delta=delta, eta=0.5, tau=0.05, w_G=0.0, w_H=w_H, psi_min=psi_min)
    out = {}
    for c in CONDS:
        curves, Hs, gd, gu = [], [], 0, 0
        for s in range(n_seeds):
            curve, H, a, b = simulate(env, c, s, mem_kw)
            curves.append(curve); Hs.append(H); gd += a; gu += b
        curves = np.array(curves)
        out[c] = dict(curves=curves, H=np.mean(Hs, axis=0), rep=curves[:, T_EXP + T_DECAY:].sum(axis=1),
                      ret=gd / max(1, gu))
    return out, sorted(env.hazard_cells)


def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def smooth(y, k=5):
    return np.convolve(y, np.ones(k) / k, mode="same")


# ---------- saved results ----------
def pct(x, ref):
    t, p = stats.ttest_rel(x, ref)
    return 100.0 * (x.mean() - ref.mean()) / ref.mean(), p


@st.cache_data(show_spinner=False)
def saved_table():
    rows, missing = [], []
    for name, path in DOMAINS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            missing.append(f"{name}: {path} was not found."); continue
        res = json.load(open(full))
        if "rep_harm_total" not in res["baseline"]["raw"][0]:
            missing.append(f"{name}: results file is missing absolute harm. Rerun it with the logging patch."); continue
        h = {c: np.array([r["rep_harm_total"] for r in res[c]["raw"]], float) for c in CONDS}
        rb, pb = pct(h["rapo"], h["baseline"]); rp, pp = pct(h["rapo"], h["pm_variant"])
        ob, po = pct(h["rapo_off_replay"], h["baseline"])
        ret = res["rapo"].get("ReplayRet", res["rapo"].get("ReplayRet_mean", float("nan")))
        rows.append(dict(domain=name, seeds=len(h["baseline"]), base=float(h["baseline"].mean()),
                         rapo=rb, p_rapo=pb, rapo_pm=rp, p_rapo_pm=pp, off=ob, p_off=po, ret=float(ret)))
    return rows, missing


def fmt_change(v, p):
    return f"{v:+.1f}% (p = {p:.2g})"


@st.cache_data(show_spinner=False)
def learned_aggregate():
    files = sorted(glob.glob(os.path.join(ROOT, "dl", "policy_table_s*_results.json")))
    if not files:
        return None
    per = []
    for f in files:
        d = json.load(open(f))
        h = {c: np.array([r["rep_harm"] for r in d["raw"][c]], float) for c in CONDS}
        per.append((pct(h["rapo"], h["baseline"])[0], pct(h["rapo"], h["pm_variant"])[0],
                    pct(h["rapo_off_replay"], h["baseline"])[0], d["summary"]["rapo"]["ReplayRet"]))
    a = np.array(per)
    return len(files), a.mean(axis=0), a.std(axis=0, ddof=1) if len(files) > 1 else np.zeros(4), a.min(axis=0), a.max(axis=0)


# ---------- hero ----------
st.markdown("""
<div class="hero">
  <h1>Stopping harm from coming back</h1>
  <p class="lede">A harmful pattern is shown, the safeguards are removed, and then the same pattern returns.
  RAPO keeps its memory in the environment, so harm stays down even after the agent has been reset.</p>
  <div class="phases">
    <div class="phase p1"><strong>Exposure</strong><span>The harmful stimulus appears and memory starts to form.</span></div>
    <div class="chev" aria-hidden="true">&rsaquo;</div>
    <div class="phase p2"><strong>Decay</strong><span>The stimulus and any intervention are removed.</span></div>
    <div class="chev" aria-hidden="true">&rsaquo;</div>
    <div class="phase p3"><strong>Replay</strong><span>The same stimulus returns. The policy is frozen.</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_live, tab_saved, tab_about = st.tabs(["Run the experiment", "Results in four domains", "How it works"])

with tab_live:
    opts = policy_options()
    st.markdown("### Settings")
    with st.form("controls"):
        c1, c2 = st.columns([2, 3], gap="large")
        policy_name = c1.selectbox("Frozen policy", list(opts.keys()),
                                   help="The agent's behaviour never changes during the experiment.")
        n_seeds = c1.slider("Random seeds", 20, 300, 80, step=20, help="More seeds give tighter confidence intervals.")
        w_H = c2.slider("Scar strength (w_H)", 0.0, 8.0, 4.0, 0.5)
        psi_min = c2.slider("Deformation floor (ψ_min). Lower means stronger suppression.", 0.02, 0.5, 0.10, 0.02)
        lam = c2.slider("Trace decay (λ)", 0.02, 0.5, 0.10, 0.02)
        delta = c2.slider("Scar persistence (δ)", 0.80, 0.995, 0.97, 0.005)
        st.form_submit_button("Run experiment")
    with st.spinner("Running Exposure, Decay and Replay..."):
        res, hazards = run_live(opts[policy_name], w_H, psi_min, lam, delta, n_seeds)

    base, rapo, pm, off = (res[c]["rep"] for c in ("baseline", "rapo", "pm_variant", "rapo_off_replay"))
    rb, pb = pct(rapo, base); rp, pp = pct(rapo, pm); ob, po = pct(off, base)

    st.markdown("### What happened in Replay")
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("RAPO vs baseline", f"{rb:+.1f}%", f"p = {pb:.2g}", delta_color="off",
              help="Change in hazard-cell visits during Replay, paired t-test across seeds.")
    m2.metric("RAPO vs fast trace only", f"{rp:+.1f}%", f"p = {pp:.2g}", delta_color="off",
              help="Does the slow, persistent scar add anything beyond a fast trace?")
    m3.metric("Deformation off at Replay", f"{ob:+.1f}%", f"p = {po:.2g}", delta_color="off",
              help="Memory is still learned, but not applied during Replay. Harm should return to baseline.")
    m4.metric("Task progress kept", f"{res['rapo']['ret']:.2f}", "1.00 = no loss", delta_color="off",
              help="How often the agent still reaches its goal, compared with no deformation.")

    left, right = st.columns(2, gap="large")
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    means = [res[c]["rep"].mean() for c in CONDS]; errs = [ci95(res[c]["rep"]) for c in CONDS]
    ax.bar(range(4), means, yerr=errs, color=[COLOR[c] for c in CONDS], capsize=4, width=0.62,
           error_kw=dict(ecolor=INK, lw=1.2))
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
    ax.set_ylabel("Hazard-cell visits in Replay"); ax.set_title("Harm in the Replay phase (95% CI)", loc="left")
    ax.grid(axis="x", visible=False)
    left.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    for (a, b), tint in zip(((0, T_EXP), (T_EXP, T_EXP + T_DECAY), (T_EXP + T_DECAY, T_EXP + T_DECAY + T_REP)), PHASE_TINT):
        ax.axvspan(a, b, color=tint, alpha=0.9, lw=0)
    for c in CONDS:
        ax.plot(smooth(res[c]["curves"].mean(axis=0)), label=LABEL[c], color=COLOR[c], lw=2)
    ax.set_xlim(0, T_EXP + T_DECAY + T_REP - 1)
    ymax = ax.get_ylim()[1]
    for x, t in ((T_EXP / 2, "Exposure"), (T_EXP + T_DECAY / 2, "Decay"), (T_EXP + T_DECAY + T_REP / 2, "Replay")):
        ax.text(x, ymax * 0.98, t, ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_xlabel("Step"); ax.set_ylabel("Mean harm per step"); ax.grid(False)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_title("Harm over the three phases", loc="left")
    right.pyplot(fig); plt.close(fig)

    st.markdown("### Where the memory forms")
    st.caption("Average scar strength after Exposure and Decay. Blue outlines mark the hazard wall. S is the start, G is the goal.")
    sage_map = LinearSegmentedColormap.from_list("sage", [PAPER, "#bcd0b4", SAGE, INK])
    fig, ax = plt.subplots(figsize=(3.7, 3.5))
    im = ax.imshow(res["rapo"]["H"].reshape(8, 8), cmap=sage_map)
    for (r, c) in hazards:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=SKY, lw=2))
    ax.text(0, 0, "S", color="w", ha="center", va="center", fontweight="bold")
    ax.text(7, 7, "G", color="w", ha="center", va="center", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, fraction=0.046)
    hm_col, note_col = st.columns([1, 2], gap="large")
    hm_col.pyplot(fig); plt.close(fig)
    note_col.markdown(
        "<div class='callout'><strong>Reading this.</strong> Scars build up on and around the hazard wall during Exposure. "
        "In Replay they steer the agent away from those cells. The fast-trace-only variant has no persistent scar, "
        "and it uses its own fixed settings. The sliders control RAPO and the ablation only.</div>",
        unsafe_allow_html=True)

with tab_saved:
    st.markdown("### Harm in Replay, change from baseline")
    st.caption("Precomputed experiments. Absolute replay harm is the primary metric. The ratio metric AUC-R is not used "
               "to compare conditions, because RAPO also lowers the Exposure harm it is divided by.")
    rows, missing = saved_table()
    if rows:
        st.dataframe(pd.DataFrame([{
            "Domain": r["domain"], "Seeds": r["seeds"], "Baseline harm": f"{r['base']:.2f}",
            "RAPO vs baseline": fmt_change(r["rapo"], r["p_rapo"]),
            "RAPO vs fast trace": fmt_change(r["rapo_pm"], r["p_rapo_pm"]),
            "Deformation off vs baseline": fmt_change(r["off"], r["p_off"]),
            "Progress kept": f"{r['ret']:.2f}"} for r in rows]), hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 1.3 + 0.8 * len(rows)))
        y = np.arange(len(rows)); hgt = 0.26
        series = [("RAPO vs baseline", "rapo", SAGE), ("RAPO vs fast trace only", "rapo_pm", SKY),
                  ("Deformation off at Replay vs baseline", "off", SAND)]
        allv = [r[k] for r in rows for _, k, _ in series]
        for i, (lab, key, col) in enumerate(series):
            vals = [r[key] for r in rows]; ys = y + (i - 1) * hgt
            ax.barh(ys, vals, height=hgt, color=col, label=lab)
            for yy, v in zip(ys, vals):
                ax.text(v + (1.5 if v >= 0 else -1.5), yy, f"{v:+.0f}%", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_xlim(min(0, min(allv)) - 14, max(0, max(allv)) + 14)
        ax.set_yticks(y); ax.set_yticklabels([r["domain"] for r in rows]); ax.invert_yaxis()
        ax.axvline(0, color=INK, lw=1); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Change in Replay harm (negative is better)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
        st.pyplot(fig); plt.close(fig)
    for m in missing:
        st.warning(m)
    agg = learned_aggregate()
    if agg is not None:
        n, mean, sd, lo, hi = agg
        st.markdown(f"### Trained neural policies ({n} separate training runs)")
        st.table(pd.DataFrame({"Mean": [f"{mean[0]:+.1f}%", f"{mean[1]:+.1f}%", f"{mean[2]:+.1f}%", f"{mean[3]:.2f}"],
                               "SD across policies": [f"{sd[0]:.1f}", f"{sd[1]:.1f}", f"{sd[2]:.1f}", f"{sd[3]:.2f}"],
                               "Range": [f"{lo[0]:+.1f}% to {hi[0]:+.1f}%", f"{lo[1]:+.1f}% to {hi[1]:+.1f}%",
                                         f"{lo[2]:+.1f}% to {hi[2]:+.1f}%", f"{lo[3]:.2f} to {hi[3]:.2f}"]},
                              index=["RAPO vs baseline", "RAPO vs fast trace only", "Deformation off at Replay vs baseline",
                                     "Task progress kept"]))
    st.caption("Harm units differ by domain (new infections, hazard-cell visits, controversial items, downvoted comments), "
               "so compare percentages, not raw counts.")

with tab_about:
    st.markdown("### The idea")
    st.markdown(
        "Safety measures that live in the agent's policy disappear when the agent is reset. RAPO keeps the memory in the "
        "environment instead. It tracks two signals for every region: a **fast trace** of recent harm and a **slow scar** "
        "of repeated harm. Both lower the chance of moving into harmful regions, but never below a fixed floor, so the "
        "change to the original system stays bounded.")
    st.markdown("### The four conditions")
    st.markdown(
        "- **Baseline:** no memory.\n"
        "- **Fast trace only:** the trace, without the persistent scar.\n"
        "- **RAPO-off@Replay:** memory is learned as usual, but switched off only during Replay. This isolates whether "
        "the effect comes from active deformation.\n"
        "- **RAPO:** trace and scar, active throughout.")
    st.markdown("### What the evidence supports")
    st.markdown(
        "- RAPO lowers absolute replay harm in all four domains, by roughly 46% to 89% against baseline.\n"
        "- Switching deformation off at Replay brings harm back to baseline in every domain.\n"
        "- The slow scar beats the fast trace alone in MovieLens and the grid-world, but not in Reddit.\n"
        "- The cost to task performance is small, except in graph diffusion (about 40%).")
    st.markdown("### Limits")
    st.markdown(
        "The policy is frozen. Environments are constructed, not production systems. Hyperparameters were tuned on the same "
        "domains they are reported on, apart from the grid-world, which used separate tuning seeds. False scars and "
        "adversarial inputs were not tested.")

st.markdown(f"<div class='site-footer'>Deep Learning course project. Code and results: "
            f"<a href='{REPO_URL}'>{REPO_URL.replace('https://', '')}</a></div>", unsafe_allow_html=True)