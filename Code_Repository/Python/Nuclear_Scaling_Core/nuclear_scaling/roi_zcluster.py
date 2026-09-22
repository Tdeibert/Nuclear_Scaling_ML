"""Recover per-nucleus z-columns from hand-drawn Fiji nucleus ROIs.

The ROI sets encode (t, z, polygon) but not nucleus identity, so `z_eq`,
`z_offset` and the equatorial band cannot be read off them directly. This module
reconstructs identity by linking ROIs through z within a timepoint, validates
each resulting column, and exposes per-(t, z, centroid) z labels for the patch
importer.

Design notes that are not obvious from the code:

* Links are matched **one-to-one between adjacent planes**, so a column visits
  each plane at most once and cannot contain two ROIs at the same z. Anton traces
  exactly one ROI per nucleus per plane, so a same-plane pair is always a merge of
  two distinct nuclei; making it unrepresentable is stronger than detecting it
  afterwards, and it removes the need to tune a neighbourhood radius.
* Candidate links are **gated** on overlap-of-the-smaller-mask and **ranked** by
  IoU. The gate admits a small cap section sitting inside the next, larger
  section (IoU alone would reject it); the ranking resolves the case where two
  nuclei stack vertically in one droplet, because the true continuation has the
  higher IoU and wins the match.
* A column that fails validation still yields per-plane mask labels; only its
  z labels are withheld. Never assert a z target that was not measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None
from skimage.draw import polygon as _sk_polygon


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _poly_bbox(poly):
    """(r0, c0, r1, c1) integer bounds of an (N,2) xy polygon."""
    return (int(np.floor(poly[:, 1].min())), int(np.floor(poly[:, 0].min())),
            int(np.ceil(poly[:, 1].max())) + 1, int(np.ceil(poly[:, 0].max())) + 1)


def _poly_mask(poly, box):
    r0, c0, r1, c1 = box
    h, w = max(r1 - r0, 1), max(c1 - c0, 1)
    if cv2 is not None:
        m = np.zeros((h, w), np.uint8)
        pts = np.round(np.column_stack([poly[:, 0] - c0, poly[:, 1] - r0])).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        return m.astype(bool)
    m = np.zeros((h, w), bool)
    rr, cc = _sk_polygon(poly[:, 1] - r0, poly[:, 0] - c0, shape=(h, w))
    m[rr, cc] = True
    return m


def _poly_area_px(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _overlap(box_a, mask_a, box_b, mask_b):
    """(intersection_px, overlap_of_smaller, iou) for two bbox-local masks."""
    r0 = max(box_a[0], box_b[0]); c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2]); c1 = min(box_a[3], box_b[3])
    if r1 <= r0 or c1 <= c0:
        return 0, 0.0, 0.0
    sa = mask_a[r0 - box_a[0]:r1 - box_a[0], c0 - box_a[1]:c1 - box_a[1]]
    sb = mask_b[r0 - box_b[0]:r1 - box_b[0], c0 - box_b[1]:c1 - box_b[1]]
    inter = int(np.count_nonzero(sa & sb))
    if inter == 0:
        return 0, 0.0, 0.0
    na, nb = int(mask_a.sum()), int(mask_b.sum())
    return inter, inter / max(min(na, nb), 1), inter / max(na + nb - inter, 1)


# ----------------------------------------------------------------------------
# config access (works with PipelineConfig or any object; all optional)
# ----------------------------------------------------------------------------
_DEFAULTS = dict(
    z_cluster_min_overlap=0.5,      # gate: intersection / smaller mask
    z_cluster_min_iou=0.10,         # floor; ranking does the real work
    z_cluster_min_planes=3,         # need 3 planes for a peak to mean anything
    z_cluster_max_wander_frac=1.0,  # centroid drift, in equatorial radii
    z_cluster_zspan_tol=1.5,        # observed span vs sphere prediction
    z_cluster_max_interior_peaks=1,
    z_cluster_require_interior_eq=True,
    equatorial_band_planes=1,
)


def _p(cfg, name):
    v = getattr(cfg, name, None)
    return _DEFAULTS[name] if v is None else v


# ----------------------------------------------------------------------------
# linking
# ----------------------------------------------------------------------------
def _link_timepoint(idx, polys, zs, cfg):
    """Greedy one-to-one adjacent-plane matching. Returns {roi_index: column_id}."""
    min_ov = _p(cfg, "z_cluster_min_overlap")
    min_iou = _p(cfg, "z_cluster_min_iou")

    boxes = {i: _poly_bbox(polys[i]) for i in idx}
    masks = {i: _poly_mask(polys[i], boxes[i]) for i in idx}
    cents = {i: (polys[i][:, 1].mean(), polys[i][:, 0].mean()) for i in idx}
    radii = {i: np.sqrt(_poly_area_px(polys[i]) / np.pi) for i in idx}

    by_z = {}
    for i in idx:
        by_z.setdefault(int(zs[i]), []).append(i)

    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for z in sorted(by_z):
        a_list, b_list = by_z.get(z), by_z.get(z + 1)
        if not b_list:
            continue
        cand = []
        for a in a_list:
            ay, ax = cents[a]
            for b in b_list:
                by, bx = cents[b]
                if np.hypot(ay - by, ax - bx) > radii[a] + radii[b]:
                    continue
                _, ov, iou = _overlap(boxes[a], masks[a], boxes[b], masks[b])
                if ov >= min_ov and iou >= min_iou:
                    cand.append((iou, ov, a, b))
        # highest-IoU links win; one match per ROI on each side keeps every
        # column single-valued at every plane.
        cand.sort(key=lambda r: (-r[0], -r[1]))
        used_a, used_b = set(), set()
        for iou, ov, a, b in cand:
            if a in used_a or b in used_b:
                continue
            used_a.add(a); used_b.add(b)
            parent[find(a)] = find(b)

    return {i: find(i) for i in idx}


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def interior_peaks(areas_by_z):
    """Local maxima strictly inside a column's area profile. A clean nucleus has
    at most one; more than one usually means two nuclei were merged."""
    v = np.asarray(areas_by_z)
    if len(v) < 3:
        return 0
    return int(((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])).sum())


def _validate(col, cfg):
    """col: DataFrame with z, area_px, cy, cx for one column. -> (ok, z_eq, reasons)."""
    reasons = []
    a = col.groupby("z").area_px.sum().sort_index()
    z_eq = int(a.idxmax())
    r_eq_px = float(np.sqrt(a.max() / np.pi))

    if len(a) < _p(cfg, "z_cluster_min_planes"):
        reasons.append("too_few_planes")

    if interior_peaks(a) > _p(cfg, "z_cluster_max_interior_peaks"):
        reasons.append("multi_peaked")

    if _p(cfg, "z_cluster_require_interior_eq") and len(a) > 1 \
            and z_eq in (int(a.index.min()), int(a.index.max())):
        # the true equator may lie outside the traced run, which would bias every
        # z_offset in the column toward zero
        reasons.append("eq_at_run_end")

    wander = float(np.hypot(col.cy - col.cy.mean(), col.cx - col.cx.mean()).max())
    if wander > _p(cfg, "z_cluster_max_wander_frac") * r_eq_px:
        reasons.append("centroid_wander")

    px = getattr(cfg, "pixel_size_um", None)
    zs_um = getattr(cfg, "z_step_um", None)
    if px and zs_um:
        # a sphere of equatorial radius r_eq spans 2*r_eq micrometres in z
        predicted = 2.0 * r_eq_px * px / zs_um
        span = int(a.index.max() - a.index.min()) + 1
        if span > _p(cfg, "z_cluster_zspan_tol") * max(predicted, 1.0):
            reasons.append("zspan_implausible")

    if int(a.index.max() - a.index.min()) + 1 != len(a):
        reasons.append("gap_in_run")

    return (not reasons), z_eq, r_eq_px, reasons


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
class ZColumns:
    """Per-nucleus z-columns and the z labels derived from them.

    `lookup` maps (t, z) -> list of (cy, cx, r_px, nucleus_id) so the importer can
    resolve a traced nucleus by its centroid without re-deriving identity.
    """

    def __init__(self, table, columns, cfg):
        self.table = table            # one row per ROI
        self.columns = columns        # one row per nucleus_id
        self.cfg = cfg
        self._band = int(_p(cfg, "equatorial_band_planes"))
        self._lookup = {}
        for r in table.itertuples():
            self._lookup.setdefault((int(r.t), int(r.z)), []).append(
                (float(r.cy), float(r.cx), float(r.r_px), int(r.nucleus_id)))
        self._by_id = {int(r.nucleus_id): r for r in columns.itertuples()}

    # -- queries ----------------------------------------------------------
    def nucleus_at(self, t, z, cy, cx, tol_frac=1.0):
        """nucleus_id whose ROI centroid at (t, z) is nearest (cy, cx), or None."""
        best, best_d = None, None
        for ry, rx, r_px, nid in self._lookup.get((int(t), int(z)), ()):
            d = np.hypot(ry - cy, rx - cx)
            if d <= tol_frac * max(r_px, 1.0) and (best_d is None or d < best_d):
                best, best_d = nid, d
        return best

    def z_labels_for(self, t, z, cy, cx):
        """-> (z_offset_um, is_equatorial, supervised) for the nucleus at (t,z,cy,cx)."""
        nid = self.nucleus_at(t, z, cy, cx)
        if nid is None:
            return 0.0, False, False
        return self.z_labels_for_id(nid, z)

    def z_labels_for_id(self, nucleus_id, z):
        row = self._by_id.get(int(nucleus_id))
        if row is None or not row.validated:
            return 0.0, False, False
        dz = abs(int(z) - int(row.z_eq))
        step = getattr(self.cfg, "z_step_um", 1.0) or 1.0
        return float(dz * step), bool(dz <= self._band), True

    @property
    def n_validated(self):
        return int(self.columns.validated.sum())

    def validated_ids(self):
        return set(self.columns.loc[self.columns.validated, "nucleus_id"].astype(int))

    # -- reporting --------------------------------------------------------
    def summary(self):
        c = self.columns
        rej = (c.loc[~c.validated, "reasons"].str.split(";").explode()
               .value_counts().to_dict()) if (~c.validated).any() else {}
        return dict(n_columns=len(c), n_validated=int(c.validated.sum()),
                    n_rois=len(self.table),
                    validated_rois=int(self.table.nucleus_id.isin(self.validated_ids()).sum()),
                    rejected_reasons=rej)

    def write_flags(self, path):
        self.columns.to_csv(path, index=False)
        return path


def assign_z(roi_items, cfg, verbose=True):
    """Group nucleus ROIs into per-nucleus z-columns and derive z labels.

    roi_items: the list returned by load_roiset(), filtered to nucleus ROIs.
    Returns a ZColumns.
    """
    items = [it for it in roi_items
             if it.get("cls", "nucleus_interior") == "nucleus_interior"]
    if not items:
        raise ValueError("no nucleus ROIs given to assign_z()")

    polys = [np.asarray(it["poly"], float) for it in items]
    ts = np.array([int(it["t"]) for it in items])
    zs = np.array([int(it["z"]) for it in items])
    areas = np.array([_poly_area_px(p) for p in polys])

    rows = []
    next_id = 0
    for t in np.unique(ts):
        idx = np.nonzero(ts == t)[0].tolist()
        comp = _link_timepoint(idx, polys, zs, cfg)
        remap = {}
        for i in idx:
            root = comp[i]
            if root not in remap:
                remap[root] = next_id
                next_id += 1
            rows.append(dict(roi_index=i, t=int(t), z=int(zs[i]),
                             cy=float(polys[i][:, 1].mean()),
                             cx=float(polys[i][:, 0].mean()),
                             area_px=float(areas[i]),
                             r_px=float(np.sqrt(areas[i] / np.pi)),
                             nucleus_id=remap[root]))

    table = pd.DataFrame(rows)

    col_rows = []
    for nid, g in table.groupby("nucleus_id"):
        ok, z_eq, r_eq_px, reasons = _validate(g, cfg)
        col_rows.append(dict(nucleus_id=int(nid), t=int(g.t.iloc[0]),
                             n_planes=int(g.z.nunique()),
                             z_lo=int(g.z.min()), z_hi=int(g.z.max()),
                             z_eq=int(z_eq), r_eq_px=round(r_eq_px, 1),
                             cy=round(float(g.cy.mean()), 1),
                             cx=round(float(g.cx.mean()), 1),
                             validated=bool(ok), reasons=";".join(reasons)))
    columns = pd.DataFrame(col_rows)

    zc = ZColumns(table, columns, cfg)
    if verbose:
        s = zc.summary()
        print(f"z-columns: {s['n_columns']} from {s['n_rois']} nucleus ROIs | "
              f"validated {s['n_validated']} ({s['n_validated']/max(s['n_columns'],1):.1%}) "
              f"covering {s['validated_rois']} ROIs "
              f"({s['validated_rois']/max(s['n_rois'],1):.1%})")
        if s["rejected_reasons"]:
            print("  rejected:", s["rejected_reasons"])
        dup = table.groupby(["nucleus_id", "z"]).size()
        assert int((dup > 1).sum()) == 0, "a column holds two ROIs at one plane"
        print("  invariant OK: no column holds two ROIs at the same plane")
    return zc
