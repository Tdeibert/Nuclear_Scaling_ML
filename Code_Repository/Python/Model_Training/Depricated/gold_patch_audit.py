# ============================================================
# E6. Completed gold-patch audit (read-only)
# ============================================================
# Plots six label-diverse patches at each of two timepoints and summarizes the
# complete metadata plus a stratified sample of label/weight arrays. Writes nothing.
from IPython.display import display

GOLD_AUDIT_TIMEPOINTS = (2, 9)
GOLD_AUDIT_PATCHES_PER_T = 6
GOLD_AUDIT_STATS_PER_T = 300
GOLD_AUDIT_SEED = 42


def _ga_pool_index(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"{root}: manifest status is {manifest.get('status')!r}")
    maps = {}
    for folder, prefix in (("images", "img_"), ("labels", "lab_"), ("weights", "wgt_")):
        maps[folder] = {p.name[len(prefix):-4]: p for p in (root / folder).glob(prefix + "*.npy")}
    keys = set(maps["images"])
    if not keys or keys != set(maps["labels"]) or keys != set(maps["weights"]):
        raise RuntimeError("Image/label/weight triplets are missing or orphaned")
    metadata = pd.read_csv(root / "patch_metadata.csv")
    if metadata.stem.duplicated().any() or set(metadata.stem) != keys:
        raise RuntimeError("patch_metadata.csv does not match file triplets")
    return manifest, metadata, maps


def _ga_stratified_rows(metadata, t, n, seed):
    frame = metadata[metadata.t == t]
    if frame.empty:
        raise ValueError(f"No gold patches for t={t}")
    rng = np.random.default_rng(seed + int(t))
    groups = list(frame.groupby(["sample_class", "z"], sort=True))
    chosen = []
    while groups and len(chosen) < min(n, len(frame)):
        next_groups = []
        for _, group in groups:
            remaining = group.loc[~group.index.isin(chosen)]
            if remaining.empty:
                continue
            chosen.append(int(rng.choice(remaining.index)))
            if len(chosen) >= min(n, len(frame)):
                break
            if len(remaining) > 1:
                next_groups.append((None, remaining))
        groups = next_groups
    return metadata.loc[chosen].copy()


def _ga_patch_stats(row, maps):
    stem = row.stem
    label = np.load(maps["labels"][stem], mmap_mode="r")
    weight = np.load(maps["weights"][stem], mmap_mode="r")
    result = dict(stem=stem, t=int(row.t), z=int(row.z), sample_class=row.sample_class)
    for head in HEAD_NAMES:
        i = HEAD_INDEX[head]
        values = np.asarray(label[..., i])
        annotated = values != UNANNOTATED
        if head == "z_offset":
            positive = annotated & (values > 0)
            result[f"{head}_median_um"] = (
                float(np.median(values[positive]) / Z_OFFSET_SCALE) if positive.any() else np.nan)
            result[f"{head}_max_um"] = (
                float(values[positive].max() / Z_OFFSET_SCALE) if positive.any() else np.nan)
        else:
            result[f"{head}_positive_frac"] = float((values == 1).mean())
        result[f"{head}_unknown_frac"] = float((~annotated).mean())
        result[f"{head}_supervised_frac"] = float((weight[..., i] > 0).mean())
    source = np.asarray(label[..., DROPLET_SOURCE_IDX])
    result["droplet_source_roi_frac"] = float((source == DSRC_ROI).mean())
    result["droplet_source_npc_frac"] = float((source == DSRC_NPC).mean())
    nucleus = np.asarray(label[..., HEAD_INDEX["nucleus_interior"]]) == 1
    result["nucleus_components"] = int(measure.label(nucleus).max())
    result["nucleus_area_um2"] = float(nucleus.sum() * cfg.pixel_size_um ** 2)
    result["mean_active_weight"] = float(weight[weight > 0].mean()) if (weight > 0).any() else 0.0
    return result


def _ga_choose_examples(rows, stats, n):
    """Greedy coverage of positive heads, source types, and sample classes."""
    table = rows.merge(stats, on=["stem", "t", "z", "sample_class"], how="inner")
    features = {}
    for _, r in table.iterrows():
        f = {"class:" + str(r.sample_class)}
        for head in HEAD_NAMES:
            if head == "z_offset":
                if np.isfinite(r.get("z_offset_max_um", np.nan)):
                    f.add("positive:z_offset")
            elif r.get(f"{head}_positive_frac", 0) > 0:
                f.add("positive:" + head)
            if r.get(f"{head}_unknown_frac", 0) > 0:
                f.add("unknown:" + head)
        if r.droplet_source_roi_frac > 0:
            f.add("source:roi")
        if r.droplet_source_npc_frac > 0:
            f.add("source:npc")
        features[r.stem] = f
    uncovered = set().union(*features.values()) if features else set()
    selected = []
    while len(selected) < min(n, len(table)):
        candidates = table[~table.stem.isin(selected)]
        if candidates.empty:
            break
        def score(row):
            new = features[row.stem] & uncovered
            # Rare positive heads outrank merely new sample classes.
            return (sum(x.startswith("positive:") for x in new), len(new),
                    float(row.nucleus_area_um2), -int(row.z))
        best = max((r for _, r in candidates.iterrows()), key=score)
        selected.append(best.stem)
        uncovered -= features[best.stem]
    return table.set_index("stem").loc[selected].reset_index(), uncovered


def _ga_overlay(ax, raw, values, title, positive_color="cyan"):
    ax.imshow(raw, cmap="gray", vmin=0, vmax=1)
    positive = values == 1
    unknown = values == UNANNOTATED
    if positive.any() and not positive.all():
        ax.contour(positive, [0.5], colors=positive_color, linewidths=0.8)
    elif positive.all():
        ax.text(.02, .98, "all positive", transform=ax.transAxes, va="top", color=positive_color)
    if unknown.any() and not unknown.all():
        ax.contour(unknown, [0.5], colors="red", linewidths=0.55)
    elif unknown.all():
        ax.text(.02, .90, "all unknown", transform=ax.transAxes, va="top", color="red")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _ga_plot_patch(row, maps, manifest, cfg=cfg):
    stem = row.stem
    image = np.load(maps["images"][stem])
    label = np.load(maps["labels"][stem])
    weight = np.load(maps["weights"][stem])
    center = image[..., (cfg.n_z_context // 2) * cfg.n_channels:
                         (cfg.n_z_context // 2 + 1) * cfg.n_channels]
    nls, npc_raw, membrane = center[..., 0], center[..., 1], center[..., 2]
    composite = np.stack([nls, membrane, npc_raw], axis=-1)
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), constrained_layout=True)
    for ax, raw, title in zip(axes[0], (nls, npc_raw, membrane, composite),
                              ("NLS", "NPC", "Membrane", "NLS/Mem/NPC")):
        ax.imshow(raw, cmap="gray" if raw.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(title); ax.axis("off")
    raw_for = dict(background=npc_raw, droplet_interior=npc_raw, droplet_edge=npc_raw,
                   npc=npc_raw, nucleus_interior=nls, nucleus_edge=nls,
                   nucleus_equatorial=nls, abnormal_nucleus=nls)
    positions = [(1,0,"background"),(1,1,"droplet_interior"),(1,2,"droplet_edge"),
                 (2,0,"npc"),(2,1,"nucleus_interior"),(2,2,"nucleus_edge"),
                 (2,3,"abnormal_nucleus"),(3,0,"nucleus_equatorial")]
    for r, c, head in positions:
        _ga_overlay(axes[r,c], raw_for[head], label[..., HEAD_INDEX[head]], head)
    source = label[..., DROPLET_SOURCE_IDX]
    axes[1,3].imshow(source, vmin=0, vmax=2,
                     cmap=ListedColormap(["black", "gold", "deepskyblue"]))
    axes[1,3].set_title("droplet_source\nblack=none gold=ROI blue=NPC", fontsize=9)
    axes[1,3].axis("off")
    z = label[..., HEAD_INDEX["z_offset"]]
    zshow = np.ma.masked_where(z == UNANNOTATED, z.astype(float) / Z_OFFSET_SCALE)
    axes[3,1].imshow(nls, cmap="gray", vmin=0, vmax=1)
    im = axes[3,1].imshow(zshow, cmap="viridis", alpha=.65)
    axes[3,1].set_title("z_offset (um; unknown hidden)", fontsize=9); axes[3,1].axis("off")
    fig.colorbar(im, ax=axes[3,1], fraction=.046)
    supervised = (weight > 0).mean(axis=-1)
    imw = axes[3,2].imshow(supervised, vmin=0, vmax=1, cmap="magma")
    axes[3,2].set_title("fraction of heads supervised", fontsize=9); axes[3,2].axis("off")
    fig.colorbar(imw, ax=axes[3,2], fraction=.046)
    lines = [f"class: {row.sample_class}", f"center: T{int(row.t)} z{int(row.z)}",
             f"xy: ({int(row.cx)}, {int(row.cy)})", f"patch: {image.shape[0]} x {image.shape[1]}",
             f"N components: {int(row.nucleus_components)}",
             f"N area: {row.nucleus_area_um2:.1f} um2",
             f"active weight mean: {row.mean_active_weight:.3f}",
             f"N unknown: {100*row.nucleus_interior_unknown_frac:.1f}%",
             f"NPC unknown: {100*row.npc_unknown_frac:.1f}%",
             f"drop unknown: {100*row.droplet_interior_unknown_frac:.1f}%",
             f"ROI drop source: {100*row.droplet_source_roi_frac:.1f}%",
             f"NPC drop source: {100*row.droplet_source_npc_frac:.1f}%"]
    axes[3,3].axis("off")
    axes[3,3].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle(stem + f"\nmanifest {manifest['status']}, hash {manifest['gold_hash']}", fontsize=11)
    plt.show(); plt.close(fig)


def audit_gold_patches(timepoints=GOLD_AUDIT_TIMEPOINTS,
                       patches_per_t=GOLD_AUDIT_PATCHES_PER_T,
                       stats_per_t=GOLD_AUDIT_STATS_PER_T,
                       root=None, cfg=cfg):
    root = Path(root) if root is not None else cfg.reviewed_root
    manifest, metadata, maps = _ga_pool_index(root)
    if int(manifest.get("n_patches", -1)) != len(metadata):
        raise RuntimeError("Manifest patch count does not match metadata")
    if (metadata.z < cfg.z_floor).any():
        raise RuntimeError(f"Found centers below z_floor={cfg.z_floor}")
    print(f"Gold pool: {len(metadata):,} complete triplets | z={metadata.z.min()}-{metadata.z.max()}")
    print("\nFull-pool patches by timepoint and class:")
    display(pd.crosstab(metadata.t, metadata.sample_class, margins=True))
    print("\nFull-pool centers by timepoint and z:")
    display(pd.crosstab(metadata.t, metadata.z))
    sampled, examples, missing = [], {}, {}
    for t in timepoints:
        rows = _ga_stratified_rows(metadata, t, stats_per_t, GOLD_AUDIT_SEED)
        stats = pd.DataFrame([_ga_patch_stats(r, maps) for _, r in rows.iterrows()])
        sampled.append(stats)
        examples[t], missing[t] = _ga_choose_examples(rows, stats, patches_per_t)
    sampled = pd.concat(sampled, ignore_index=True)
    summary = []
    for t, group in sampled.groupby("t"):
        for head in HEAD_NAMES:
            row = dict(t=t, head=head, n_patches=len(group),
                       median_unknown=group[f"{head}_unknown_frac"].median(),
                       median_supervised=group[f"{head}_supervised_frac"].median())
            if head != "z_offset":
                row["patches_with_positive"] = int((group[f"{head}_positive_frac"] > 0).sum())
                row["median_positive_fraction"] = group[f"{head}_positive_frac"].median()
            else:
                row["patches_with_positive"] = int(group.z_offset_max_um.notna().sum())
                row["median_positive_fraction"] = np.nan
            summary.append(row)
    summary = pd.DataFrame(summary)
    print(f"\nPixel statistics from {len(sampled):,} stratified patches:")
    display(summary.round(4))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    pd.crosstab(metadata.t, metadata.sample_class).plot(kind="bar", stacked=True, ax=axes[0])
    axes[0].set_title("All patches by sample class"); axes[0].set_ylabel("patches")
    sampled.groupby("t").nucleus_area_um2.median().plot(kind="bar", ax=axes[1], color="seagreen")
    axes[1].set_title("Median positive nucleus area"); axes[1].set_ylabel("um2")
    unknown_cols=[f"{h}_unknown_frac" for h in HEAD_NAMES]
    sampled.groupby("t")[unknown_cols].mean().T.plot(kind="bar", ax=axes[2])
    axes[2].set_title("Mean unknown fraction by head"); axes[2].set_ylabel("fraction")
    axes[2].set_xticklabels(HEAD_NAMES, rotation=75, ha="right")
    plt.tight_layout(); plt.show(); plt.close(fig)
    for t in timepoints:
        print(f"\nT{t}: selected {len(examples[t])} label-diverse examples")
        if missing[t]:
            print("Features not represented in six selected patches:", sorted(missing[t]))
        for _, row in examples[t].iterrows():
            _ga_plot_patch(row, maps, manifest, cfg)
    return dict(manifest=manifest, metadata=metadata, sampled_stats=sampled,
                summary=summary, examples=examples, missing_features=missing)


GOLD_PATCH_AUDIT = audit_gold_patches((2, 9), patches_per_t=6, stats_per_t=300)
