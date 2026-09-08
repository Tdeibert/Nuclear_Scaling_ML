"""
vulcan_review.py -- one-keypress-per-candidate review of artifact candidates.

Run in napari_env on StarForge:

    python vulcan_review.py /path/to/review_dir

Loads review_stack.tif (N, Z, C=[NLS, NPC, Mem], Y, X) and review_index.csv,
shows one candidate at a time (centre z-plane, all planes scrollable), and
writes decisions.csv incrementally so a crashed session loses nothing.

Keys
  a  spurious: no nucleus in this droplet at this plane      -> negative
  c  cap / off-equator section of a real nucleus             -> negative
  f  fragment / under-segmentation of a real nucleus         -> positive, relabel
  r  real nucleus, segmentation fine                         -> skipped (counts toward tier precision)
  u  unsure                                                  -> skipped
  backspace  undo last decision       n / p  next / previous without deciding

This is a classification review, not a mask review: the model needs to know
WHAT the candidate was, and the 2.0 labeler supplies the boundary for 'f'.
"""
import sys, csv
from pathlib import Path
import numpy as np, pandas as pd, tifffile, napari

review_dir = Path(sys.argv[1]).expanduser()
stack = tifffile.imread(review_dir / "review_stack.tif")          # (N, Z, C, Y, X)
idx = pd.read_csv(review_dir / "review_index.csv")
dec_path = review_dir / "decisions.csv"
done = pd.read_csv(dec_path).set_index("review_index")["decision"].to_dict() if dec_path.exists() else {}
N, Z, C = stack.shape[:3]
order = [i for i in range(N) if i not in done] + [i for i in range(N) if i in done]
pos = 0

v = napari.Viewer(title="Vulcan candidate review")
zc = Z // 2
names = ["NLS", "NPC", "Membrane"]; cmaps = ["magenta", "green", "gray"]
layers = [v.add_image(stack[order[0], :, c], name=names[c], colormap=cmaps[c], blending="additive",
                      contrast_limits=np.percentile(stack[order[0], zc, c], [1, 99.8])) for c in range(C)]
v.dims.set_point(0, zc)

def show(i):
    r = idx.iloc[i]
    for c in range(C):
        layers[c].data = stack[i, :, c]
        layers[c].contrast_limits = np.percentile(stack[i, zc, c], [1, 99.8])
    v.dims.set_point(0, zc)
    v.title = (f"[{pos+1}/{N}] idx {i}  t={int(r.time_frame)} z={int(round(r.centroid_z_px))}  "
               f"tier {int(r.tier)}  area {r.cross_sectional_area_um2:.0f} um2  N/C {r.nc_fraction:.2f}  "
               f"sig import/repair/persist={int(r.sig_import)}{int(r.sig_repair)}{int(r.sig_persist)}  "
               f"decision: {done.get(i, '-')}")

def save():
    with open(dec_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["review_index", "decision"])
        for k in sorted(done): w.writerow([k, done[k]])

def decide(key):
    global pos
    done[order[pos]] = key; save()
    if pos < N - 1: pos += 1
    show(order[pos])

for key in "acfru":
    v.bind_key(key, (lambda k: (lambda viewer: decide(k)))(key))
@v.bind_key("n")
def _next(viewer):
    global pos; pos = min(pos + 1, N - 1); show(order[pos])
@v.bind_key("p")
def _prev(viewer):
    global pos; pos = max(pos - 1, 0); show(order[pos])
@v.bind_key("Backspace")
def _undo(viewer):
    global pos; pos = max(pos - 1, 0); done.pop(order[pos], None); save(); show(order[pos])

show(order[0])
print(f"{len(done)}/{N} already decided; decisions -> {dec_path}")
napari.run()
