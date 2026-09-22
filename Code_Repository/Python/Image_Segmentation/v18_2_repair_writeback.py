# =============================================================================
# v18.2 patch -- carry repaired measurements into grouped_z_df
#
# The problem, precisely: repair_fragmented_nuclei takes grouped_z_df as its
# second parameter and never reads it. The name appears exactly once in the
# function -- in the signature. The write-back was intended and never landed.
#
# Consequences in the reference export (control_extract_1.1):
#   * 81 nuclei have a Selected_Slice_ID absent from NucleusZStack, because the
#     z-sweep re-selected onto a plane that produced no original detection.
#   * 181 nucleus/plane pairs report different areas in Nuclei.csv and
#     NucleusZStack.csv for the same nucleus at the same plane.
#   * Every repaired nucleus's per-plane rows are pre-repair, so NucleusZStack
#     describes the fragmented core (~1/3 of true area per Section 30b), not
#     the repaired nucleus. This affects far more rows than the 81.
#
# The per-plane repaired measurements already exist -- repair_plane returns
# them for every plane in the sweep -- they were just discarded in favour of
# the maximum. This patch collects them and merges them back.
#
# APPLY: replace repair_fragmented_nuclei in cell 76, replace cell 78, and
# apply the one-line change to cell 134 shown at the bottom.
# =============================================================================


# --- cell 76: replacement for repair_fragmented_nuclei -----------------------

def merge_repaired_planes(grouped_z_df: pd.DataFrame,
                          plane_records: list,
                          verbose: bool = True) -> pd.DataFrame:
    """Fold per-plane repaired measurements back into grouped_z_df.

    Planes that already existed are updated in place; planes the repair
    measured for the first time are appended. Provenance is kept in
    `repair_source` so a downstream consumer can tell an original detection
    from a repaired one, and `area_px_original` preserves what was replaced.

    Returns a new frame. grouped_z_df is not mutated -- the caller decides
    what to persist, which matters because the raw grouping is a cache that
    must stay re-runnable.
    """
    if grouped_z_df is None or grouped_z_df.empty:
        return grouped_z_df
    out = grouped_z_df.copy()
    if "repair_source" not in out.columns:
        out["repair_source"] = "original"
    if "area_px_original" not in out.columns:
        out["area_px_original"] = out["area_px"]
    if not plane_records:
        return out

    rec = pd.DataFrame(plane_records)
    rec = (rec.sort_values(["t", "nucleus_3d_id", "z", "area_px"])
              .drop_duplicates(["t", "nucleus_3d_id", "z"], keep="last"))

    key = ["t", "nucleus_3d_id", "z"]
    out = out.set_index(key)
    rec = rec.set_index(key)

    hit = rec.index.intersection(out.index)
    new = rec.index.difference(out.index)

    # -- update planes that already existed
    if len(hit):
        out.loc[hit, "area_px"] = rec.loc[hit, "area_px"].astype(int)
        for c in ("centroid_x_px", "centroid_y_px"):
            upd = rec.loc[hit, c]
            out.loc[hit, c] = upd.where(upd.notna(), out.loc[hit, c])
        out.loc[hit, "repair_source"] = "repaired"

    # -- append planes the repair measured for the first time
    if len(new):
        add = rec.loc[new].reset_index()
        for c in out.reset_index().columns:
            if c not in add.columns:
                add[c] = np.nan
        add["class_name"] = "nucleus"
        add["area_px_original"] = np.nan       # there was no original
        add["repair_source"] = "repaired_new_plane"
        add = add[out.reset_index().columns]
        out = pd.concat([out.reset_index(), add], ignore_index=True).set_index(key)

    out = out.reset_index()
    if verbose:
        print(f"  grouped_z write-back: {len(hit)} planes updated, "
              f"{len(new)} planes added "
              f"({len(grouped_z_df)} -> {len(out)} rows)")
    return out


def repair_fragmented_nuclei(best_z_df, grouped_z_df, image_5d,
                             nucleus_instance_4d, droplet_instance_4d,
                             config, verbose=True):
    """
    Detect fragmented nuclei, repair them, re-select z, and return BOTH the
    corrected best-Z table and a grouped_z table carrying the repaired
    per-plane measurements.

    Returns (best_z_df, grouped_z_df). The second return value is new in
    v18.2 -- previously the repaired per-plane measurements were computed
    during the z-sweep and thrown away, leaving grouped_z_df (and therefore
    NucleusZStack) describing the unrepaired fragments.

    Overwrites `area_px`, `z`, and the centroids with the repaired values so
    every downstream stage consumes them unchanged. Originals are preserved as
    `*_original`, and provenance goes in `repair_status` / `repair_gain` /
    `ratio_vs_npc`.
    """
    px2 = config.pixel_size_um ** 2
    H, W = image_5d.shape[-2], image_5d.shape[-1]
    Z = image_5d.shape[1]
    em = max(int(round(config.repair_edge_margin_um / config.pixel_size_um)), 1)

    d = best_z_df.copy().reset_index(drop=True)
    d["area_px_original"] = d.area_px
    d["z_original"] = d.z
    d["nucleus_area_um2"] = d.area_px * px2

    plane_records = []          # NEW: every plane the repair actually measured

    # ---- edge status: droplet OR nucleus touching the frame ----
    def touches(arr, lab):
        if not lab:
            return True
        ys, xs = np.where(arr == lab)
        return bool(ys.size == 0 or ys.min() <= em or xs.min() <= em
                    or ys.max() >= H - 1 - em or xs.max() >= W - 1 - em)

    edge, sol = [], []
    for r in d.itertuples():
        inst = np.asarray(nucleus_instance_4d[int(r.t), int(r.z)])
        dpl = np.asarray(droplet_instance_4d[int(r.t), int(r.z)])
        nl = _resolve_instance(inst, r.centroid_x_px, r.centroid_y_px)
        dl = _resolve_instance(dpl, r.centroid_x_px, r.centroid_y_px)
        edge.append(touches(inst, nl) or touches(dpl, dl))
        if nl:
            rp = measure.regionprops((inst == nl).astype(np.uint8))[0]
            sol.append(float(rp.solidity))
        else:
            sol.append(np.nan)
    d["exclude_edge"] = edge
    d["solidity_original"] = sol

    pool = d[~d.exclude_edge]
    ref = pool.groupby("t").nucleus_area_um2.quantile(0.75).rename("_ref")
    d = d.join(ref, on="t")
    d["area_deficit"] = 1.0 - d.nucleus_area_um2 / d._ref
    d["is_fragmented"] = ((d.solidity_original < config.repair_min_solidity)
                          | (d.area_deficit > config.repair_area_deficit))
    d = d.drop(columns=["_ref"])

    for c in ("repair_gain", "ratio_vs_npc", "npc_r_um", "npc_area_um2",
              "npc_n_rays", "z_shift", "n_planes_repaired"):
        d[c] = np.nan
    d["repair_status"] = "not fragmented"

    if verbose:
        print(f"{len(d)} nuclei | {int(d.exclude_edge.sum())} at frame edge "
              f"| {int(d.is_fragmented.sum())} fragmented "
              f"({100*d.is_fragmented.mean():.1f}%)")

    todo = d.index[d.is_fragmented & ~d.exclude_edge]
    n_ok = n_fail = 0
    for i in todo:
        r = d.loc[i]
        t, z0 = int(r.t), int(r.z)
        n3d = r.nucleus_3d_id
        cx, cy = float(r.centroid_x_px), float(r.centroid_y_px)
        mask0, m0, off0 = repair_plane(t, z0, cx, cy, image_5d,
                                       nucleus_instance_4d, droplet_instance_4d,
                                       config)
        if not mask0.any():
            d.at[i, "repair_status"] = f"rejected: {m0['reject']}"
            n_fail += 1
            continue

        def _record(z_, m_):
            """Keep a repaired plane measurement for the grouped_z write-back."""
            plane_records.append({
                "t": t, "nucleus_3d_id": n3d, "z": int(z_),
                "area_px": int(m_["area_px"]),
                "centroid_x_px": m_.get("centroid_x_px", np.nan),
                "centroid_y_px": m_.get("centroid_y_px", np.nan),
            })

        _record(z0, m0)

        best, best_z, best_mask, best_off, seen = m0, z0, mask0, off0, 1
        for direction in (-1, +1):
            seed = mask0
            for step in range(1, config.repair_z_pad + 1):
                z = z0 + direction * step
                if not (0 <= z < Z):
                    break
                sd = morphology.binary_erosion(seed, morphology.disk(3))
                if not sd.any():
                    break
                mk, mm, off = repair_plane(t, z, cx, cy, image_5d,
                                           nucleus_instance_4d,
                                           droplet_instance_4d, config,
                                           seed_override=sd)
                if not mk.any() or mm["area_px"] < 0.15 * best["area_px"]:
                    break
                seen += 1
                _record(z, mm)                      # NEW
                if mm["area_px"] > best["area_px"]:
                    best, best_z, best_mask, best_off = mm, z, mk, off
                seed = mk

        d.at[i, "area_px"] = best["area_px"]
        d.at[i, "nucleus_area_um2"] = best["area_um2"]
        d.at[i, "z"] = best_z
        d.at[i, "z_shift"] = best_z - z0
        d.at[i, "n_planes_repaired"] = seen
        if "centroid_x_px" in best:
            d.at[i, "centroid_x_px"] = best["centroid_x_px"]
            d.at[i, "centroid_y_px"] = best["centroid_y_px"]
        d.at[i, "repair_gain"] = best["area_px"] / max(r.area_px, 1)
        d.at[i, "npc_r_um"] = best["npc_r_um"]
        d.at[i, "npc_area_um2"] = best["npc_area_um2"]
        d.at[i, "npc_n_rays"] = best["npc_n_rays"]
        if np.isfinite(best["npc_area_um2"]) and best["npc_area_um2"] > 0:
            d.at[i, "ratio_vs_npc"] = best["area_um2"] / best["npc_area_um2"]
        d.at[i, "repair_status"] = "repaired"

        # ── persist the winning mask (full-frame) so halo/radial-sweep find it ──
        r0, c0 = best_off
        canvas = np.zeros((H, W), dtype=bool)
        bh, bw = best_mask.shape
        canvas[r0:r0 + bh, c0:c0 + bw] = best_mask
        out_path = _repaired_mask_path(config, t, r.nucleus_3d_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, canvas)

        n_ok += 1

    if verbose:
        rep = d[d.repair_status == "repaired"]
        print(f"  repaired {n_ok} | rejected {n_fail}")
        if n_fail:
            print(d.loc[todo, "repair_status"][
                d.loc[todo, "repair_status"].str.startswith("rejected")]
                .value_counts().head(5).to_string())
        if len(rep):
            print(f"  median gain {rep.repair_gain.median():.2f}x | "
                  f"z moved {int((rep.z_shift != 0).sum())}/{len(rep)} | "
                  f"median planes {rep.n_planes_repaired.median():.0f}")
            print(f"  ratio vs NPC ring {rep.ratio_vs_npc.median():.3f} "
                  f"(target 0.88-0.95)")
            if rep.ratio_vs_npc.median() > 1.0:
                print("  WARNING: above 1.0 — the mask is crossing the "
                      "envelope. Lower cfg.repair_npc_slack.")
            elif rep.ratio_vs_npc.median() < 0.85:
                print("  NOTE: below 0.85 — still under-recovering. Consider "
                      "raising cfg.repair_flatten_um.")

    grouped_z_repaired = merge_repaired_planes(grouped_z_df, plane_records, verbose)
    return d, grouped_z_repaired


# --- cell 78: replacement runner ---------------------------------------------
"""
repaired_path         = cfg.obj_dir / "best_z_nuclei_repaired.pkl"
grouped_repaired_path = cfg.obj_dir / "grouped_z_objects_repaired.pkl"
RUN_FRAGMENTATION_REPAIR = True

if cfg.repair_enabled and RUN_FRAGMENTATION_REPAIR:
    require_segmentation(cfg, needs=("nucleus_instance_hyperstack",
                                     "droplet_instance_hyperstack"))
    t0 = time.perf_counter()
    nucleus_instance_4d = tiff.memmap(cfg.nucleus_instance_hyperstack_path, mode="r")
    droplet_instance_4d = tiff.memmap(cfg.droplet_instance_hyperstack_path, mode="r")
    best_z_df, grouped_z_df = repair_fragmented_nuclei(
        best_z_df, grouped_z_df, img_5d,
        nucleus_instance_4d, droplet_instance_4d, cfg)
    best_z_df.to_pickle(repaired_path)
    grouped_z_df.to_pickle(grouped_repaired_path)
    print(f"Fragmentation repair finished in {time.perf_counter() - t0:.2f}s")
elif cfg.repair_enabled:
    best_z_df   = pd.read_pickle(repaired_path)
    grouped_z_df = pd.read_pickle(grouped_repaired_path)
    print("Loaded repaired best-Z and grouped-Z tables.")
else:
    print("cfg.repair_enabled is False — using unrepaired tables.")

print(best_z_df.shape, grouped_z_df.shape)
display(best_z_df.head())
"""

# --- cell 134: one line, so the export records provenance --------------------
"""
Inside build_zstack_table's row loop, add the source column:

        rows.append({
            ...
            "Is_Selected_Max":          bool(row.z == row.best_z),
            "Repair_Source":            getattr(row, "repair_source", "original"),
        })

and add "Repair_Source" to `cols`. build_zstack_table needs no other change --
it already reads grouped_z_df, which now carries the repaired values.
"""
