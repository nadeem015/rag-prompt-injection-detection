"""Publication figures (IEEE column widths) generated ONLY from files in results/."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "final"; F = R / "figures"; F.mkdir(exist_ok=True, parents=True)
SRC = ROOT / "results" / "final_experiment"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, ORANGE, AQUA, VIOLET, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#8a8985"
LABEL = {"word_tfidf": "Word TF-IDF", "word_char_tfidf": "Word+char TF-IDF", "rule_only": "Rules (frozen)"}
COL = {"word_tfidf": BLUE, "word_char_tfidf": ORANGE, "rule_only": AQUA}
LS = {"word_tfidf": "-", "word_char_tfidf": "--", "rule_only": ":"}
MK = {"word_tfidf": "o", "word_char_tfidf": "s", "rule_only": "^"}

def save(fig, name):
    fig.savefig(F / f"{name}.png", bbox_inches="tight"); fig.savefig(F / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 1: architecture ---------------------------------------------------------
def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(ax, p, q, c=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=c))

fig, ax = plt.subplots(figsize=(7.1, 2.55)); ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1); ax.axis("off")
box(ax, .005, .66, .105, .22, "User\nquery", "#f4f4f2", GREY)
box(ax, .15, .60, .135, .34, "Layer 1\nQuery screening\n(rule / ML / hybrid)", "#fdf0ea", ORANGE)
box(ax, .325, .66, .115, .22, "TF-IDF\nretriever\n(top-$k$ chunks)")
box(ax, .48, .60, .145, .34, "Layer 2\nChunk screening\nscore + explain", "#fdf0ea", ORANGE)
box(ax, .665, .66, .13, .22, "Allowed\nchunks")
box(ax, .83, .60, .16, .34, "Layer 3\nBoundary-safe prompt\n(delimiters neutralised)", "#e9f7f1", AQUA)
box(ax, .48, .20, .145, .24, "Quarantined\nchunks (excluded)", "#fbe9e9", "#c0392b")
box(ax, .15, .20, .135, .24, "Blocked query:\nrefusal message", "#fbe9e9", "#c0392b")
box(ax, .83, .20, .16, .24, "Extractive answer\n(no generative LLM)", "#f4f4f2", GREY)
box(ax, .15, .02, .475, .12, "Layer 4: append-only JSONL security-event log", "#eeeaf9", VIOLET)
arrow(ax, (.11, .77), (.15, .77)); arrow(ax, (.285, .77), (.325, .77)); arrow(ax, (.44, .77), (.48, .77))
arrow(ax, (.625, .77), (.665, .77)); arrow(ax, (.795, .77), (.83, .77)); arrow(ax, (.91, .60), (.91, .44))
arrow(ax, (.217, .60), (.217, .44), "#c0392b"); arrow(ax, (.552, .60), (.552, .44), "#c0392b")
for x in (.217, .552): arrow(ax, (x, .20), (x, .14), VIOLET)
ax.text(.225, .52, "flagged", fontsize=6, color="#c0392b"); ax.text(.56, .52, "flagged", fontsize=6, color="#c0392b")
ax.text(.74, .915, "clean", fontsize=6, color=INK, ha="center")
ax.text(.005, .30, "Untrusted:\nuser query and\nretrieved text", fontsize=6.3, color=GREY, va="center")
save(fig, "fig1_architecture")

# ---------------- Fig 2: confusion matrices ----------------------------------------------------
sm = pd.read_csv(SRC / "final_summary_metrics.csv"); sm = sm[sm.dataset == "agentic5k_test"].set_index("model")
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
fig.subplots_adjust(wspace=0.08)
for ax, m in zip(axs, ["word_tfidf", "word_char_tfidf"]):
    r = sm.loc[m]; M = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=300)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8, color="white" if M[i, j] > 150 else INK)
    ax.set_xticks([0, 1], ["Benign", "Injection"]); ax.set_yticks([0, 1], ["Benign", "Injection"])
    ax.set_title(f"{LABEL[m]}\n($t$={r.threshold:.2f})", fontsize=7); ax.set_xlabel("Predicted")
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel("Actual")
save(fig, "fig2_confusion")

# ---------------- Fig 3: ROC + PR (headline) ---------------------------------------------------
cv = json.load(open(R / "curves" / "curve_data.json")); auc = pd.read_csv(R / "metrics" / "auc_ap.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.55))
for key, m in [("word_tfidf", "word_tfidf"), ("word_char_tfidf", "word_char_tfidf"), ("rule_score", "rule_only")]:
    c = cv[f"agentic5k_test|{key}"]; a = auc[(auc.scope == "agentic5k_test") & (auc.scorer == key)].iloc[0]
    axs[0].plot(c["fpr"], c["tpr"], color=COL[m], ls=LS[m], label=f"{LABEL[m]} (AUC {a.roc_auc:.3f})")
    axs[1].plot(c["recall"], c["precision"], color=COL[m], ls=LS[m], label=f"{LABEL[m]} (AP {a.avg_precision:.3f})")
axs[0].plot([0, 1], [0, 1], color=GREY, lw=.6, ls=(0, (1, 2)))
for m, thr in [("word_tfidf", .47), ("word_char_tfidf", .60)]:
    r = sm.loc[m]; axs[0].scatter(r.fp / (r.fp + r.tn), r.tp / (r.tp + r.fn), s=22, color=COL[m], marker=MK[m], zorder=5, edgecolor="white", lw=.6)
    axs[1].scatter(r.tp / (r.tp + r.fn), r.precision, s=22, color=COL[m], marker=MK[m], zorder=5, edgecolor="white", lw=.6)
axs[1].axhline(280 / 545, color=GREY, lw=.6, ls=(0, (1, 2)))
axs[0].set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="(a) ROC, 545-example test split")
axs[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall, same split", ylim=(0.4, 1.02))
h, l = axs[0].get_legend_handles_labels(); l = [x.split(" (")[0] for x in l]
fig.legend(h, [f"{a} (AUC {auc[(auc.scope=='agentic5k_test')&(auc.scorer==k)].roc_auc.iloc[0]:.3f}; AP {auc[(auc.scope=='agentic5k_test')&(auc.scorer==k)].avg_precision.iloc[0]:.3f})" for a, k in zip(l, ["word_tfidf","word_char_tfidf","rule_score"])], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -0.08))
fig.tight_layout(rect=(0, .04, 1, 1)); save(fig, "fig3_roc_pr")

# ---------------- Fig 4: per-dataset F1 -------------------------------------------------------
am = pd.read_csv(R / "metrics" / "all_detectors_all_scopes.csv")
order = ["agentic5k_test", "boundary_pairs_test", "neuralchemy_test", "slabs_test", "synthetic_docs_test", "synthetic_direct_test", "synthetic_v2_test", "external_diagnostic_pooled"]
names = ["Agentic5K\ntest", "boundary\npairs", "neuralchemy", "slabs", "synthetic\ndocs", "synthetic\ndirect", "synthetic\nv2", "Pooled\ndiag."]
fig, ax = plt.subplots(figsize=(7.1, 2.55)); w = .26
for k, m in enumerate(["word_tfidf", "word_char_tfidf", "rule_only"]):
    v = [am[(am.scope == s) & (am.detector == m)].f1.iloc[0] for s in order]
    b = ax.bar(np.arange(len(order)) + (k - 1) * w, v, w * .92, color=COL[m], label=LABEL[m], hatch=["", "///", "..."][k], edgecolor="white", lw=.4)
    for x, y in zip(np.arange(len(order)) + (k - 1) * w, v): ax.text(x, y + .012, f"{y:.2f}", ha="center", fontsize=5.2, rotation=90, va="bottom", zorder=8, bbox=dict(fc="white", ec="none", pad=0.3, alpha=.85))
af = [am[(am.scope == s) & (am.detector == "always_flag")].f1.iloc[0] for s in order]
ax.scatter(np.arange(len(order)), af, marker="_", s=420, color=INK, lw=1.4, zorder=3, label="Always-flag baseline")
ax.set_xticks(np.arange(len(order)), names); ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.08)
ax.legend(ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.13)); ax.grid(axis="y", lw=.3, color="#dddcd8"); ax.set_axisbelow(True)
save(fig, "fig4_per_dataset_f1")

# ---------------- Fig 5: validation threshold sweep ---------------------------------------------
sw = pd.read_csv(SRC / "validation_threshold_sweep.csv")
fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.0), sharex=True)
for m in ["word_tfidf", "word_char_tfidf"]:
    s = sw[sw.model == m]; thr = {"word_tfidf": .47, "word_char_tfidf": .60}[m]
    axs[0].plot(s.threshold, s.f1, color=COL[m], ls=LS[m], label=LABEL[m]); axs[1].plot(s.threshold, s.fpr, color=COL[m], ls=LS[m])
    for ax, col in zip(axs, ["f1", "fpr"]):
        ax.scatter([thr], [s[np.isclose(s.threshold, thr)][col].iloc[0]], s=20, color=COL[m], marker=MK[m], zorder=5, edgecolor="white", lw=.6)
axs[1].axhline(.10, color=INK, lw=.6, ls=(0, (3, 2))); axs[1].text(.55, .14, "FPR $\\leq$ 0.10 constraint", fontsize=6)
axs[0].set(ylabel="Validation F1", title="Validation sweep (467 examples); markers = selected $t$"); axs[1].set(ylabel="Validation FPR", xlabel="Decision threshold $t$")
axs[0].legend(frameon=False, loc="lower center")
save(fig, "fig5_threshold_sweep")

# ---------------- Fig 6: context screening simulation -----------------------------------------
cs = pd.read_csv(R / "metrics" / "context_screening_simulation_summary.csv"); cs = cs[cs.position == "suffix"].set_index("detector")
dets = ["rule", "word_tfidf", "word_char_tfidf", "hybrid_or_word_char_tfidf", "hybrid_fusion_word_char_tfidf"]
lab = ["Rules", "Word", "Word+char", "Hybrid-OR\n(W+C)", "Hybrid-fusion\n(W+C)"]
fig, ax = plt.subplots(figsize=(3.45, 2.2)); x = np.arange(len(dets)); w = .36
e = [cs.loc[d, "exposure_mean"] for d in dets]; q = [cs.loc[d, "false_quarantine_mean"] for d in dets]
ax.bar(x - w / 2, e, w * .92, color=BLUE, label="Injection payload reaching prompt", edgecolor="white")
ax.bar(x + w / 2, q, w * .92, color=ORANGE, hatch="///", label="Benign composite chunks quarantined", edgecolor="white")
for xi, v in zip(x - w / 2, e): ax.text(xi, v + .015, f"{v:.2f}", ha="center", fontsize=5.6)
for xi, v in zip(x + w / 2, q): ax.text(xi, v + .015, f"{v:.2f}", ha="center", fontsize=5.6)
ax.set_xticks(x, lab, fontsize=5.8); ax.set_ylim(0, 1.0); ax.set_ylabel("Rate (mean of 3 seeds)")
ax.legend(frameon=False, loc="upper left", fontsize=5.8, bbox_to_anchor=(0, 1.2))
save(fig, "fig6_context_screening")
print("figures:", sorted(p.name for p in F.glob("*.png")))
