"""All 30-epoch validation curves in one axes.

hue = tokenizer class, marker = model within class, so two models sharing a hue
are still separable without color alone.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_GLOB = sys.argv[1] if len(sys.argv) > 1 else "logs/train30-*.out"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/val_curves_30ep_single"

CLASS = {
    "6-mer non-overlapping": ("#2a78d6", ["ntv2", "ntv1"]),
    "nucleotide":            ("#eb6834", ["aidodna", "hyenadna_large", "caduceus", "evo2"]),
    "BPE":                   ("#1baf7a", ["seqlens", "genalm", "grover", "mistraldna", "dnabert2"]),
    "6-mer overlapping":     ("#4a3aa7", ["dnabert1"]),
}
MARKERS = ["o", "^", "s", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

EPOCH_RE = re.compile(r"epoch=(\d+).*?val_loss=([\d.]+)")
BEST_RE = re.compile(r"best_val_loss=([\d.]+)")

runs = {}
for path in sorted(glob.glob(LOG_GLOB)):
    with open(path) as fh:
        lines = fh.readlines()
    if not lines:
        continue
    parts = lines[0].split()
    model = parts[2] if len(parts) > 2 else os.path.basename(path)
    ep, va, best = [], [], None
    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            ep.append(int(m.group(1)))
            va.append(float(m.group(2)))
        b = BEST_RE.search(line)
        if b:
            best = float(b.group(1))
    if ep:
        runs[model] = (ep, va, best)

if not runs:
    sys.exit("no runs parsed from %s" % LOG_GLOB)

fig, ax = plt.subplots(figsize=(10, 6.2))
offset = 0
for cname, (color, members) in CLASS.items():
    for k, model in enumerate(members):
        if model not in runs:
            continue
        ep, va, best = runs[model]
        ax.plot(ep, va, color=color, lw=2, marker=MARKERS[k % len(MARKERS)],
                ms=8, markevery=(offset % 4, 4), mec="white", mew=1.5,
                label="%s  (%s)" % (model, cname), zorder=3)
        offset += 1

ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(labelsize=10, colors=INK2, length=0)
ax.set_xlabel("epoch", fontsize=11, color=INK2)
ax.set_ylabel("validation loss", fontsize=11, color=INK2)
ax.set_title("Validation loss, 30-epoch budget\nhue = tokenizer class, marker = model",
             fontsize=13, color=INK, loc="left", pad=12)
ax.legend(fontsize=9, frameon=False, labelcolor=INK2,
          loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig("%s.%s" % (OUT, ext), dpi=200, facecolor="#fcfcfb",
                bbox_inches="tight")
    print("wrote %s.%s" % (OUT, ext))

print("")
print("%-18s %8s %8s %14s" % ("model", "best", "final", "still falling?"))
for model in sorted(runs, key=lambda m: runs[m][2] or 9e9):
    ep, va, best = runs[model]
    tail = va[-4:]
    print("%-18s %8.4f %8.4f %14s" % (
        model, best or float("nan"), va[-1],
        "YES" if tail[-1] < tail[0] - 1e-4 else "no"))
