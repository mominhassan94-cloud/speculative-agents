"""Render the system diagram (diagram.png). Same graph as diagram.mmd, drawn with matplotlib
because no Graphviz/Mermaid renderer was available in the build environment.

  pip install -r reports/status_update/requirements.txt
  python reports/status_update/make_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, RED, ORANGE, GREY = "#dbe8f7", "#dcefd9", "#f6dcdc", "#fbe9cf", "#eeeeee"

fig, ax = plt.subplots(figsize=(11, 4.9))
ax.set_xlim(0, 11)
ax.set_ylim(0.75, 5.65)
ax.axis("off")

def box(x, y, w, h, text, color, bold_first=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444", lw=1))
    lines = text.split("\n")
    if bold_first:
        ax.text(x, y + (0.1 if len(lines) > 1 else 0), lines[0], ha="center", va="center",
                fontsize=8.6, weight="bold")
        if len(lines) > 1:
            ax.text(x, y - 0.13, "\n".join(lines[1:]), ha="center", va="center", fontsize=7.2, color="#333")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=7.6)


def arrow(p, q, text=None, dashed=False, rad=0.0, tx=None, ty=None, color="#333"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color,
                linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        ax.text(tx if tx is not None else (p[0] + q[0]) / 2, ty if ty is not None else (p[1] + q[1]) / 2,
                text, fontsize=7, ha="center", va="center", color=color,
                bbox=dict(fc="white", ec="none", pad=0.5))


# --- main agent loop (left) ---------------------------------------------------------------
box(1.6, 5.25, 2.4, 0.45, "User task in\nrun_agent(task, brain, runtime)", GREY)
box(4.45, 5.25, 2.5, 0.5, "Final answer + metrics out\ntime, hits, misses, wasted, too-late", GREY)
box(2.3, 4.3, 3.0, 0.55, "Agent LLM call (Gemini function calling)\nGeminiBrain.next_action: history -> next tool call", BLUE)
box(2.3, 3.25, 3.3, 0.55, "Runtime.call: real call == a pending guess?\nexact match on tool name + args", ORANGE)
box(1.25, 2.2, 1.9, 0.55, "HIT\nreuse cached result", GREEN)
box(3.4, 2.2, 2.1, 0.55, "MISS / no guess\ndiscard guesses, run real tool", RED)
box(2.3, 1.2, 2.7, 0.5, "Observation appended to history\nthen Runtime.observe() fires speculator", GREY)

arrow((1.6, 5.02), (1.6, 4.58))
arrow((3.3, 4.58), (3.75, 5.0), "final text", tx=3.15, ty=4.82)
arrow((2.3, 4.02), (2.3, 3.53), "tool call", tx=2.75)
arrow((1.75, 2.97), (1.35, 2.48), "yes", tx=1.4, ty=2.75)
arrow((2.85, 2.97), (3.3, 2.48), "no", tx=3.25, ty=2.75)
arrow((1.25, 1.92), (1.9, 1.45))
arrow((3.4, 1.92), (2.75, 1.45))
ax.plot([0.95, 0.35, 0.35], [1.2, 1.2, 4.3], color="#333", lw=1.1)
arrow((0.35, 4.3), (0.8, 4.3))
ax.text(0.35, 2.75, "loop", fontsize=7, ha="center", va="center", rotation=90,
        bbox=dict(fc="white", ec="none", pad=0.5))

# --- speculative path (right), runs in a background thread pool ------------------------------
ax.add_patch(FancyBboxPatch((5.75, 0.82), 5.15, 3.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="none", ec="#999", lw=1, linestyle="--"))
ax.text(8.3, 4.55, "Speculative path: background threads, overlaps the next Gemini call",
        fontsize=7.6, ha="center", style="italic", color="#555")
box(7.65, 1.2, 3.5, 0.55, "Speculator predicts next tool call\nRuleSpeculator (instant) or LLMSpeculator (Gemini)", BLUE)
box(7.35, 2.35, 2.9, 0.6, "Safety gate\nread-only?  confidence >= 0.6?\nagent not already moved on?", ORANGE)
box(9.85, 2.35, 1.8, 0.6, "Drop guess\nblocked / held back\n/ too late", GREY)
box(7.35, 3.6, 2.6, 0.55, "Run tool early\nthread pool -> pending cache", GREEN)

arrow((3.65, 1.2), (5.9, 1.2), "history so far", tx=4.8, ty=1.32)
arrow((7.35, 1.48), (7.35, 2.05))
arrow((8.8, 2.35), (8.95, 2.35))
ax.text(8.87, 2.75, "no", fontsize=7, ha="center")
arrow((7.35, 2.65), (7.35, 3.32), "yes", tx=7.6, ty=2.98)
arrow((6.05, 3.6), (3.95, 3.3), dashed=True, color="#2a6d2a")
ax.text(4.95, 3.72, "pending result checked at\nthe agent's next real call", fontsize=7, ha="center",
        color="#2a6d2a")

plt.tight_layout(pad=0.2)
out = os.path.join(HERE, "diagram.png")
plt.savefig(out, dpi=200)
print("wrote", out)
