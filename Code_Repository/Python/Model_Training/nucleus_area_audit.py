"""Drop-in cross-sectional area audit for the H4 segmentation test."""

from IPython.display import display


def plot_nuclear_cross_sections(cfg=cfg, detections=None, seed=42):
    """Plot one selected best-Z cross-section for every linked nucleus."""
    if detections is None:
        if "SEGTEST" in globals() and isinstance(SEGTEST, dict):
            detections = SEGTEST.get("detections")
        if detections is None:
            p = cfg.qc_dir / "segtest_detections.csv"
            if not p.exists():
                raise FileNotFoundError(f"{p} not found; run H4 first")
            detections = pd.read_csv(p)
    d = detections.copy()
    if d.empty or "is_best_z" not in d:
        raise ValueError("No best-Z detections are available")
    if d.is_best_z.dtype == object:
        d["is_best_z"] = d.is_best_z.astype(str).str.lower().isin(["true", "1"])
    best = (d[d.is_best_z].sort_values("area_um2", ascending=False)
            .drop_duplicates(["t", "nucleus_id"]).copy())
    best["physical_flag"] = False
    if "rho0" in best:
        lo, hi = cfg.rho0_impossible_window
        best["physical_flag"] |= best.rho0.between(lo, hi, inclusive="both")
    if "chord_ok" in best:
        chord = (best.chord_ok.astype(str).str.lower().isin(["true", "1"])
                 if best.chord_ok.dtype == object else best.chord_ok.astype(bool))
        best["physical_flag"] |= ~chord

    q = best.groupby("t").area_um2
    stats = q.agg(n="size", mean="mean", median="median", std="std",
                  minimum="min", q25=lambda x:x.quantile(.25),
                  q75=lambda x:x.quantile(.75), maximum="max")
    stats["cv"] = stats["std"] / stats["mean"]
    stats["iqr"] = stats.q75 - stats.q25
    stats["physical_flags"] = best.groupby("t").physical_flag.sum()
    stats = stats.reset_index()

    # Same spatial holdout and same one-best-Z/equatorial-section criterion for gold.
    gold = pd.DataFrame()
    zpath = cfg.reviewed_root / "z_cluster_flags.csv"
    mpath = cfg.qc_dir / "segtest_prob.json"
    if zpath.exists() and mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf8"))
        y0, x0, y1, x1 = meta["box"]
        zc = pd.read_csv(zpath)
        valid = zc.validated.astype(str).str.lower().isin(["true", "1"])
        inside = zc.cy.between(y0, y1, inclusive="neither") & zc.cx.between(x0, x1, inclusive="neither")
        source_ok = zc.source.eq("gold") if "source" in zc else True
        gold = zc[valid & inside & source_ok & zc.t.isin(meta["t_range"])].copy()
        # r_eq_px is the equivalent radius of the manually traced best-Z ROI.
        gold["area_um2"] = np.pi * (gold.r_eq_px * cfg.pixel_size_um) ** 2
    gold_stats = pd.DataFrame()
    if len(gold):
        gq = gold.groupby("t").area_um2
        gold_stats = gq.agg(n="size", mean="mean", median="median", std="std",
                            minimum="min", q25=lambda x:x.quantile(.25),
                            q75=lambda x:x.quantile(.75), maximum="max")
        gold_stats["cv"] = gold_stats["std"] / gold_stats["mean"]
        gold_stats["iqr"] = gold_stats.q75 - gold_stats.q25
        gold_stats = gold_stats.reset_index()

    times = sorted(best.t.astype(int).unique())
    rng = np.random.default_rng(seed)
    fig, (ax, trend) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={"height_ratios":[2.2, 1]})
    for t in times:
        x = best[best.t == t]
        normal, flagged = x[~x.physical_flag], x[x.physical_flag]
        ax.scatter(t+rng.uniform(-.16,.16,len(normal)),normal.area_um2,s=20,alpha=.55,
                   color="#168aad",edgecolors="none",label="detected nucleus" if t==times[0] else None)
        ax.scatter(t+rng.uniform(-.16,.16,len(flagged)),flagged.area_um2,s=32,alpha=.9,
                   color="#d00000",marker="x",label="physical-gate flag" if t==times[0] else None)
        row=stats[stats.t==t].iloc[0]
        ax.vlines(t,row.q25,row.q75,color="black",lw=5)
        ax.scatter(t,row["median"],s=75,color="white",edgecolor="black",zorder=5,
                   label="median / IQR" if t==times[0] else None)
        gx = gold[gold.t == t]
        if len(gx):
            ax.scatter(t+rng.uniform(-.16,.16,len(gx)),gx.area_um2,s=30,alpha=.6,
                       color="#6a4c93",marker="+",linewidths=1.2,
                       label="gold nucleus" if t==times[0] else None)
    if len(gold_stats):
        ax.plot(gold_stats.t,gold_stats["median"],"D-",color="#6a4c93",lw=1.5,
                label="gold median")
    ax.axhline(cfg.gate_median_area_final_um2,color="orange",ls="--",lw=1.2,
               label=f"legacy gate ({cfg.gate_median_area_final_um2:g} µm²)")
    ax.set(title="Every linked nucleus at its selected best-Z plane\nNo manual artifact exclusions",
           xlabel="Time point",ylabel="Cross-sectional area (µm²)")
    ax.set_xticks(times); ax.grid(axis="y",alpha=.2); ax.legend(ncol=2,fontsize=9)

    trend.plot(stats.t,stats["median"],"o-",color="#168aad",label="median")
    trend.fill_between(stats.t.to_numpy(),stats.q25.to_numpy(),stats.q75.to_numpy(),
                       color="#168aad",alpha=.2,label="IQR")
    if len(gold_stats):
        trend.plot(gold_stats.t,gold_stats["median"],"D-",color="#6a4c93",label="gold median")
        trend.fill_between(gold_stats.t.to_numpy(),gold_stats.q25.to_numpy(),gold_stats.q75.to_numpy(),
                           color="#6a4c93",alpha=.15,label="gold IQR")
    trend.set(title="Area trajectory",xlabel="Time point",ylabel="Area (µm²)")
    trend.set_xticks(times); trend.grid(alpha=.2); trend.legend()
    plt.tight_layout(); plt.show()
    print(f"Best-Z nuclei plotted: {len(best)}")
    print("Red crosses are automated physical flags, not manually confirmed artifacts.")
    print("Purple crosses are validated gold nuclei from the same holdout tile at their equatorial section.")
    print("\nPredicted best-Z areas:")
    display(stats.round(2))
    if len(gold_stats):
        print("\nGold equatorial areas:")
        display(gold_stats.round(2))
    else:
        print("No validated gold columns were found in the segmentation-test holdout tile.")
    return dict(best_z=best,statistics=stats,gold=gold,gold_statistics=gold_stats)


NUCLEAR_AREA_AUDIT = plot_nuclear_cross_sections()
