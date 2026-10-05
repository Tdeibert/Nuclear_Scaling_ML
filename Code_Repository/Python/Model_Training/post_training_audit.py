"""Drop-in post-training operating-point audit for Vulcan 2.5.3.

Run after H1-H4. Reuses segtest_prob.tif; it never reruns inference or changes
models/pools. It writes compact QC tables and figures under cfg.qc_dir.
"""

from IPython.display import display


def _pa_match(lab, gold_locs, box, iou_min=P3_MATCH_IOU, keep_fp=False):
    """Exact H3 matching, with optional unmatched predicted regions."""
    y0, x0 = box[:2]
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    preds = [r for r in measure.regionprops(lab) if r.label not in border]
    pairs = []
    for gi, ((a, b, c, d), gm) in enumerate(gold_locs):
        ga = int(gm.sum())
        for r in preds:
            pa, pb, pc, pd_ = r.bbox
            ya, xa = max(a-y0, pa), max(b-x0, pb)
            yb, xb = min(c-y0, pc), min(d-x0, pd_)
            if ya >= yb or xa >= xb:
                continue
            inter = int((gm[ya-(a-y0):yb-(a-y0), xa-(b-x0):xb-(b-x0)] &
                         (lab[ya:yb, xa:xb] == r.label)).sum())
            if inter:
                pairs.append((inter/(ga+r.area-inter), gi, r.label, r.area/ga))
    ug, up, ious, ratios = set(), set(), [], []
    for iou, gi, pl, ratio in sorted(pairs, reverse=True):
        if iou < iou_min:
            break
        if gi in ug or pl in up:
            continue
        ug.add(gi); up.add(pl); ious.append(iou); ratios.append(ratio)
    out = dict(tp=len(ious), fp=len(preds)-len(up), fn=len(gold_locs)-len(ug),
               sum_iou=float(np.sum(ious)), area_ratios=ratios)
    if keep_fp:
        out["fp_regions"] = [r for r in preds if r.label not in up]
    return out


def _pa_filter_lab(lab, core, eq=None, zoff=None, mean_p=0.0, min_px=0,
                   eq_min=None, zmax=None):
    out = np.zeros_like(lab)
    for r in measure.regionprops(lab):
        m = lab[r.slice] == r.label
        if r.area < min_px or float(core[r.slice][m].mean()) < mean_p:
            continue
        if eq_min is not None and (eq is None or float(eq[r.slice][m].mean()) < eq_min):
            continue
        if zmax is not None and (zoff is None or float(np.abs(zoff[r.slice][m]).mean()) > zmax):
            continue
        out[r.slice][m] = r.label
    return out


def _pa_summary(plane_rows, keys):
    d = pd.DataFrame(plane_rows)
    g = d.groupby(keys, dropna=False).agg(tp=("tp", "sum"), fp=("fp", "sum"),
                                          fn=("fn", "sum"), sum_iou=("sum_iou", "sum"))
    g["precision"] = g.tp / (g.tp + g.fp).clip(lower=1)
    g["recall"] = g.tp / (g.tp + g.fn).clip(lower=1)
    g["f1"] = 2*g.precision*g.recall/(g.precision+g.recall).replace(0, np.nan)
    g["matched_iou"] = g.sum_iou/g.tp.clip(lower=1)
    return g.reset_index()


def _pa_pick(overall, recall_floor=0.90):
    ok = overall[overall.recall >= recall_floor]
    pool = ok if len(ok) else overall
    return pool.sort_values(["f1", "precision", "fp"], ascending=[False, False, True]).iloc[0]


def _pa_gold_gate_benchmark(gold, meta, box, cfg):
    """Gold equatorial ROI areas, one validated Z-column per biological nucleus."""
    p = cfg.reviewed_root / "z_cluster_flags.csv"
    out = {}
    if p.exists():
        zc = pd.read_csv(p)
        valid = zc.validated.astype(str).str.lower().isin(["true", "1"])
        inside = ((zc.cx > box[1]) & (zc.cx < box[3]) &
                  (zc.cy > box[0]) & (zc.cy < box[2]))
        zc = zc[valid & inside & zc.t.isin(meta["t_range"])]
        zc["gold_eq_area_um2"] = np.pi*(zc.r_eq_px*cfg.pixel_size_um)**2
        final = zc[zc.t == max(meta["t_range"])].gold_eq_area_um2
        out["gold_column_n_final"] = int(len(final))
        out["gold_area_cv_final"] = float(final.std()/final.mean()) if len(final) > 1 else np.nan
        out["gold_median_area_final_um2"] = float(final.median()) if len(final) else np.nan
        out["legacy_area_cv_gate"] = float(cfg.gate_area_cv_final)
        out["legacy_median_area_gate_um2"] = float(cfg.gate_median_area_final_um2)
        out["gold_pass_area_cv"] = bool(out["gold_area_cv_final"] <= cfg.gate_area_cv_final) if len(final)>1 else None
        out["gold_pass_median_area"] = bool(out["gold_median_area_final_um2"] >= cfg.gate_median_area_final_um2) if len(final) else None
    # Per-plane gold is reported separately; it is not equivalent to a best-Z nucleus.
    tmax = max(meta["t_range"])
    plane_areas = [float(m.sum())*cfg.pixel_size_um**2 for (t, _), locs in gold.items()
                   if t == tmax for _, m in locs]
    out["gold_plane_roi_n_final"] = len(plane_areas)
    out["gold_plane_roi_median_um2"] = float(np.median(plane_areas)) if plane_areas else np.nan
    return out


def vulcan_post_audit(prob_path=None, cfg=cfg,
                      mean_p_grid=(0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
                      min_area_um2_grid=(0, 25, 50, 75, 100, 125, 150),
                      eq_grid=(None, 0.4, 0.5, 0.6),
                      zmax_grid=(None, 2.0, 4.0, 6.0), recall_floor=0.90):
    """Audit component confidence/size and optional equatorial/Z rejection."""
    prob_path = Path(prob_path or (cfg.qc_dir / "segtest_prob.tif"))
    meta_path = prob_path.with_suffix(".json")
    if not prob_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run H4 once; segtest_prob.tif/json are required")
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    ch = {n:i for i,n in enumerate(meta["channels"])}
    for required in ("nucleus_interior", "nucleus_edge"):
        if required not in ch:
            raise RuntimeError(f"Probability volume lacks {required}")
    box = tuple(meta["box"]); mm = tiff.memmap(str(prob_path))
    hs = load_memmap_tiff(cfg.image_file)
    gold = p3_gold_in_box(box, meta["t_range"], meta["z_range"], cfg)
    px2 = cfg.pixel_size_um**2
    min_grid = tuple(float(x) for x in min_area_um2_grid)
    eq_available = "nucleus_equatorial" in ch
    z_available = "z_offset" in ch
    eq_grid = tuple(eq_grid) if eq_available else (None,)
    zmax_grid = tuple(zmax_grid) if z_available else (None,)
    print(f"Reusing {prob_path}; channels={meta['channels']}")
    print(f"Gold ROIs in holdout: {sum(map(len,gold.values()))}; no model/pool files will change")

    # Watershed once per plane. All sweeps filter those fixed production components.
    cache, sweep_rows = {}, []
    for ti,t in enumerate(meta["t_range"]):
        for zi,z in enumerate(meta["z_range"]):
            core = np.asarray(mm[ti,zi,ch["nucleus_interior"]], np.float32)
            edge = np.asarray(mm[ti,zi,ch["nucleus_edge"]], np.float32)
            eq = np.asarray(mm[ti,zi,ch["nucleus_equatorial"]], np.float32) if eq_available else None
            zo = np.asarray(mm[ti,zi,ch["z_offset"]], np.float32) if z_available else None
            nls = read_crop(hs,t,z,cfg.nucleus_channel_idx,box).astype(np.float32)
            base = watershed_nuclei(nls,core,edge,cfg) if cfg.use_watershed_postproc else measure.label(core>cfg.mask_threshold)
            cache[(ti,zi)] = (core,eq,zo,nls,base)
            for mp in mean_p_grid:
                for au in min_grid:
                    lab = _pa_filter_lab(base,core,mean_p=float(mp),min_px=int(np.ceil(au/px2)))
                    sweep_rows.append(dict(mean_p=float(mp),min_area_um2=au,t=int(t),z=int(z),
                                           **_pa_match(lab,gold.get((t,z),[]),box)))
    by_t = _pa_summary(sweep_rows,["mean_p","min_area_um2","t"])
    overall = _pa_summary(sweep_rows,["mean_p","min_area_um2"])
    t0 = by_t[by_t.t==0][["mean_p","min_area_um2","fp"]].rename(columns={"fp":"t0_fp"})
    overall = overall.merge(t0,on=["mean_p","min_area_um2"],how="left").fillna({"t0_fp":0})
    chosen = _pa_pick(overall,recall_floor)
    base_mp, base_au = float(chosen.mean_p), float(chosen.min_area_um2)

    print("\n== component confidence / minimum-area sweep ==")
    display(overall.sort_values("f1",ascending=False).head(15).round(4))
    print(f"Base recommendation: mean_p >= {base_mp:.2f}, area >= {base_au:.0f} um2; "
          f"precision={chosen.precision:.3f}, recall={chosen.recall:.3f}, F1={chosen.f1:.3f}, T0 FP={int(chosen.t0_fp)}")

    # Conditional equatorial and |z-offset| rejection at the selected operating point.
    filt_rows=[]
    for ti,t in enumerate(meta["t_range"]):
        for zi,z in enumerate(meta["z_range"]):
            core,eq,zo,nls,base=cache[(ti,zi)]
            for eqmin in eq_grid:
                for zmax in zmax_grid:
                    lab=_pa_filter_lab(base,core,eq,zo,base_mp,int(np.ceil(base_au/px2)),eqmin,zmax)
                    filt_rows.append(dict(eq_min=-1 if eqmin is None else float(eqmin),
                                          zmax_um=-1 if zmax is None else float(zmax),t=int(t),z=int(z),
                                          **_pa_match(lab,gold.get((t,z),[]),box)))
    filt_t=_pa_summary(filt_rows,["eq_min","zmax_um","t"])
    filt=_pa_summary(filt_rows,["eq_min","zmax_um"])
    t0f=filt_t[filt_t.t==0][["eq_min","zmax_um","fp"]].rename(columns={"fp":"t0_fp"})
    filt=filt.merge(t0f,on=["eq_min","zmax_um"],how="left").fillna({"t0_fp":0})
    selected=_pa_pick(filt,recall_floor)
    eq_sel=None if selected.eq_min<0 else float(selected.eq_min)
    z_sel=None if selected.zmax_um<0 else float(selected.zmax_um)
    print("\n== optional equatorial / z-offset rejection ==")
    display(filt.sort_values("f1",ascending=False).round(4))
    print(f"Final recommendation: eq_min={eq_sel}, |z_offset| max={z_sel}; "
          f"precision={selected.precision:.3f}, recall={selected.recall:.3f}, F1={selected.f1:.3f}, T0 FP={int(selected.t0_fp)}")

    final_t=filt_t[(filt_t.eq_min==selected.eq_min)&(filt_t.zmax_um==selected.zmax_um)].copy()
    print("\n== recommended operating point by timepoint ==")
    display(final_t.drop(columns="sum_iou").round(4))

    # Representative unmatched predictions, deliberately enriched for T0/T1/T2/T9.
    examples=[]
    for ti,t in enumerate(meta["t_range"]):
        if t not in (0,1,2,9): continue
        for zi,z in enumerate(meta["z_range"]):
            core,eq,zo,nls,base=cache[(ti,zi)]
            lab=_pa_filter_lab(base,core,eq,zo,base_mp,int(np.ceil(base_au/px2)),eq_sel,z_sel)
            mt=_pa_match(lab,gold.get((t,z),[]),box,keep_fp=True)
            for r in mt["fp_regions"]:
                if sum(e["t"]==t for e in examples)>=2: break
                cy,cx=map(int,r.centroid); rad=64
                a,b=max(0,cy-rad),max(0,cx-rad); c,d=min(lab.shape[0],cy+rad),min(lab.shape[1],cx+rad)
                examples.append(dict(t=t,z=z,nls=nls[a:c,b:d].copy(),core=core[a:c,b:d].copy(),
                                     mask=(lab[a:c,b:d]==r.label),area=r.area*px2,
                                     mean_p=float(core[lab==r.label].mean())))
    if examples:
        fig,axs=plt.subplots(int(np.ceil(len(examples)/2)),2,figsize=(10,4*np.ceil(len(examples)/2)))
        axs=np.atleast_1d(axs).ravel()
        for ax,e in zip(axs,examples):
            ax.imshow(e["nls"],cmap="gray"); ax.contour(e["mask"],[0.5],colors="red",linewidths=1.5)
            ax.imshow(np.ma.masked_where(e["core"]<0.25,e["core"]),cmap="cool",alpha=.25,vmin=0,vmax=1)
            ax.set_title(f"FP T{e['t']} Z{e['z']}  area={e['area']:.0f} um2  mean p={e['mean_p']:.2f}")
            ax.axis("off")
        for ax in axs[len(examples):]: ax.axis("off")
        plt.tight_layout(); plt.show()

    gates=_pa_gold_gate_benchmark(gold,meta,box,cfg)
    print("\n== legacy area gates measured on gold nuclei ==")
    display(pd.DataFrame([gates]).T.rename(columns={0:"value"}))

    qc=cfg.qc_dir; qc.mkdir(parents=True,exist_ok=True)
    overall.to_csv(qc/"postaudit_operating_grid.csv",index=False)
    by_t.to_csv(qc/"postaudit_operating_grid_by_t.csv",index=False)
    filt.to_csv(qc/"postaudit_eq_z_grid.csv",index=False)
    final_t.to_csv(qc/"postaudit_recommended_by_t.csv",index=False)
    report=dict(prob_path=str(prob_path),component_filter=dict(mean_p=base_mp,min_area_um2=base_au),
                optional_filter=dict(eq_min=eq_sel,zmax_um=z_sel),recall_floor=recall_floor,
                selected={k:(v.item() if hasattr(v,"item") else v) for k,v in selected.to_dict().items()},
                gold_gate_benchmark=gates)
    (qc/"postaudit_report.json").write_text(json.dumps(report,indent=2,default=str)+"\n",encoding="utf8")
    print("Saved post-audit tables/report to",qc)
    return dict(operating_grid=overall,by_timepoint=by_t,filter_grid=filt,
                recommended_by_timepoint=final_t,gold_gates=gates,report=report)


POST_AUDIT = vulcan_post_audit()
