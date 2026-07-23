# Nuclear Scaling Pipeline — Quantification Reference

Every measured and derived quantity, with its formula, units, and where it is
computed. Symbols are reused consistently throughout:

| symbol | meaning |
|---|---|
| $s$ | pixel size, µm/px (`pixel_size_um`) |
| $\Delta z$ | Z step, µm (`z_step_um`) |
| $N$ | pixel count |
| $A$ | cross-sectional area |
| $r$ | nuclear radius, $R$ droplet radius |
| $\theta$ | ray angle; $\rho$ normalised radius |
| $I$ | intensity |

> **Verification note.** Formulas marked **[code]** were read directly from the
> v15/v16 notebook. Those marked **[convention]** state the intended definition
> where I have not re-read the exact implementation — check these against the
> source before they go into a methods section.

---

## 1. Calibration and coordinates

| Quantity | Formula | Units | Notes |
|---|---|---|---|
| Pixel size | $s = \dfrac{p_{\text{sensor}}}{M_{\text{obj}} \times M_{\text{relay}}}$ | µm/px | $p_{\text{sensor}} = 6.5$ µm. **Unresolved:** 40× alone → 0.1625; 40× with 0.75× C-mount → 0.2167. Config previously held 0.108 (= 6.5/60), which was wrong. |
| Pixel size from a known object | $s = \dfrac{D_{\text{true}}}{d_{\text{px}}}$ | µm/px | Independent check. Droplets: $D_{\text{true}} \approx 45\text{–}50$ µm. **Must use the equatorial plane** — an off-equator chord underestimates $d_{\text{px}}$ and so overestimates $s$. |
| Area conversion | $A_{\mu m^2} = N_{\text{px}} \, s^2$ | µm² | Every area in the export. Errors in $s$ enter **squared**. |
| Length conversion | $\ell_{\mu m} = \ell_{\text{px}} \, s$ | µm | |
| Pixel threshold from a physical one | $N = A_{\mu m^2} / s^2$ | px | e.g. `min_nucleus_area_px` = 50 µm² ÷ 0.2167² = 1065 |
| True acquisition time | $T = t \, T_{\text{frame}} + i_{\text{tile}} \, \Delta t_{\text{tile}}$ | min | **[code]** $T_{\text{frame}} = n_{\text{tiles}} \Delta t_{\text{tile}} = 6 \times 1 = 6$ min. Tiles within a frame are minutes apart, so $T$ is not $6t$. |

---

## 2. Segmentation (image → probability → mask)

| Quantity | Formula | Notes |
|---|---|---|
| Per-channel normalisation | $x' = \mathrm{clip}\!\left(\dfrac{x - P_{1}}{P_{99.8} - P_{1}},\, 0,\, 1\right)$ | **[code]** Percentiles computed per channel per plane. Replaced global max-scaling at v15. |
| Patch tiling | patches at stride $S$ over a $P{\times}P$ grid | **[code]** $P = 512$, $S < P$ so patches overlap |
| Hann window | $w(y,x) = w_1(y)\,w_1(x)$, $w_1(i) = \tfrac12\!\left[1 - \cos\!\dfrac{2\pi i}{P-1}\right]$ | **[code]** Separable 2D Hann |
| Overlap blending | $\hat p(y,x) = \dfrac{\sum_k w_k(y,x)\, p_k(y,x)}{\sum_k w_k(y,x)}$ | **[code]** Weighted average over all patches covering a pixel. Prevents seams. |
| Class probability | $p_c = \sigma(z_c)$, independent per class | **[code]** Independent sigmoids, **not** softmax — so $\sum_c p_c \neq 1$ |
| Binary mask | $M_c = [\,p_c > \tau_c\,]$ | **[code]** $\tau_{\text{nuc}}=0.5$, $\tau_{\text{NPC}}=0.3$, $\tau_{\text{drop}}=0.5$ |
| Label collapse | $L = \arg\max_c$ over thresholded classes, background elsewhere | **[code]** `collapse_label_map` |
| Morphological cleaning | opening radius $\rho_o$, hole fill $\le A_h$ | **[code]** Both defined in px, so both must be rescaled when $s$ changes |

---

## 3. Object geometry (per connected component, per plane)

| Quantity | Formula | Units | Notes |
|---|---|---|---|
| Area | $A = N_{\text{px}} s^2$ | µm² | **[code]** `regionprops` area |
| Equivalent radius | $r_{\text{eq}} = \sqrt{A/\pi}$ | µm | Used for all radius-based reasoning |
| Equivalent diameter | $d_{\text{eq}} = 2\sqrt{A/\pi}$ | µm | |
| Centroid | $\bar y = \frac1N\sum y_i$, $\bar x = \frac1N\sum x_i$ | px | **[code]** Stored in px; multiply by $s$ for µm |
| Solidity | $A / A_{\text{convex hull}}$ | — | Fragmented or merged objects score low |
| Circularity | $\dfrac{4\pi A}{P^2}$ | — | $P$ = perimeter. 1 = perfect circle. |
| Eccentricity | $\sqrt{1 - (b/a)^2}$ | — | $a,b$ = major/minor axes of the equivalent ellipse |
| Volume (slice sum) | $V = \Delta z \sum_z A_z$ | µm³ | **[convention]** Riemann sum over the Z stack |
| Volume (spherical) | $V = \tfrac43 \pi r_{\text{eq}}^3$ | µm³ | Alternative; assumes sphericity. Confirm which the export uses. |
| Nucleus/droplet area fraction | $f = A_{\text{nuc}} / A_{\text{drop}}$ | — | **Calibration-free.** Measured $f \approx 0.284$ |
| Radius ratio | $r/R = \sqrt{f}$ | — | From $f = 0.284$: $r/R \approx 0.53$ |

---

## 4. Z handling

| Quantity | Formula | Notes |
|---|---|---|
| Focus window | $\mathcal{Z}_t = \{z : |z - z^*_t| \le w_f,\; z \ge z_{\min}\}$ | **[code]** $w_f$ = `focus_window_radius` (=1, so 3 planes); $z_{\min}$ = `focus_min_z` (=6). Only these planes are segmented. |
| Frame best-focus plane | $z^*_t = \arg\max_z \; \Phi(z)$ | **[code]** $\Phi$ = NPC-ring focus score, `get_best_focus_z_indices_adaptive` |
| Object-based focus score | $\Phi(z) = \sum_{\text{objects}} \mathbb{1}[A_{\min} \le A \le A_{\max}] \cdot \phi_{\text{ring}}$ | **[convention]** Counts nucleus-like objects weighted by ring sharpness. $A_{\max}$ = `focus_max_nucleus_area_um2`; if this exceeds droplet area, droplets score as nuclei. |
| Z grouping | join objects in adjacent planes if $\lVert \mathbf{c}_{z} - \mathbf{c}_{z+1}\rVert \, s < \delta_z$ | **[code]** $\delta_z$ = `z_group_tolerance_um` |
| Best-Z per nucleus | $z_{\text{sel}} = \arg\max_z A_z$ | **[code]** **Known weakness:** largest area ≠ best focus. An out-of-focus blurred object can be larger than the sharp one. |
| Cross-section of a sphere | $A(d) = \pi\!\left(r^2 - d^2\right)$ | $d$ = distance of the plane from the object centre. Basis for all cap/equator reasoning. |
| Fractional area change per Z step | $\dfrac{\Delta A}{A} = \dfrac{2d\,\Delta z + \Delta z^2}{r^2 - d^2}$ | Near-flat at the equator, steep off it. Used to confirm the Z window straddles the equator. |

---

## 5. Tracking

| Quantity | Formula | Notes |
|---|---|---|
| Search radius | $\varepsilon_{\text{px}} = \varepsilon_{\mu m} / s$ | **[code]** `track_dbscan_eps_um`. Rescales with $s$ — at the corrected calibration a 6 µm setting had been acting as 12 µm. |
| Track assignment | DBSCAN over $(\bar x s, \bar y s)$ across frames, $\varepsilon$ as above | **[code]** `assign_track_ids_hybrid_dbscan`. Labels existing rows only — never creates rows for missed detections. |
| Track span | $n_t(\text{id}) = \lvert \{t : \text{id present}\} \rvert$ | Longitudinal sample size, distinct from per-frame counts |
| Multi-nucleus exclusion | reject if a second nucleus lies within $\delta_m$ | **[code]** `multi_nucleus_exclusion_um` |

---

## 6. Intensity quantification

Halos are concentric annuli stepping outward from the nuclear boundary, each of
width $w_h$ (`halo_step_px`), sampled in each channel.

| Quantity | Formula | Notes |
|---|---|---|
| Halo annulus $k$ | $H_k = \{\,\mathbf{p} : (k-1)w_h < \mathrm{dist}(\mathbf{p}, \partial M) \le k\,w_h\,\}$ | **[convention]** Built by successive dilation/erosion of the nucleus mask. Verify direction (inward vs outward) in code. |
| Halo intensity | $I_k = \mathrm{mean}\{\,I(\mathbf{p}) : \mathbf{p} \in H_k\,\}$ | **[code]** Per channel, per halo |
| Background | $I_{\text{bg}}$ = mean over droplet-free region | **[code]** Per channel, per nucleus |
| Background-corrected | $I' = \max(I - I_{\text{bg}},\, 0)$ | Apply before any cross-timepoint intensity comparison |
| Nuclear signal | $I_{\text{nuc}}$ = mean NLS intensity inside the nucleus mask | **[code]** |
| Cytoplasmic signal | $I_{\text{cyt}}$ = mean NLS intensity in the droplet outside the nucleus | **[code]** |
| **N/C ratio** | $\mathrm{NC} = \dfrac{I_{\text{nuc}}}{I_{\text{cyt}}}$ | Unbounded, $[0,\infty)$. **Calibration-independent** — a ratio of intensities, so unaffected by $s$. |
| **N/C fraction** | $f_{NC} = \dfrac{\mathrm{NC}}{1 + \mathrm{NC}} = \dfrac{I_{\text{nuc}}}{I_{\text{nuc}} + I_{\text{cyt}}}$ | Bounded $[0,1]$. **$f_{NC} = 0.5$ means no import** — a droplet interior gives exactly 0.5 by construction. |
| Nucleus probability (mean) | $\bar p = \dfrac{1}{N}\sum_{\mathbf{p} \in M} p_{\text{nuc}}(\mathbf{p})$ | Proposed export column. Boundary pixels always pull this down. |
| Nucleus probability (p10) | 10th percentile of $p_{\text{nuc}}$ within $M$ | More informative than $\min$, which sits at $\tau$ for every object |
| Per-class confidence vector | $\bar p_c = \frac1N\sum_{\mathbf{p}\in M} p_c(\mathbf{p})$ for all $c$ | A droplet misclassified as nucleus shows elevated $\bar p_{\text{droplet}}$ |

---

## 7. Radial profile

Rays are cast from the nucleus centroid outward to the droplet wall at
$n_\theta$ angles. The nucleus interior is excluded, so each ray begins at the
mask boundary.

| Quantity | Formula | Units | Notes |
|---|---|---|---|
| Ray angles | $\theta_j = 360 j / n_\theta$ | deg | **[code]** `radial_n_angles` |
| Sample distance | $d_i$ = distance from centroid along $\theta$ | px | **[code]** `distance_px`, step `RAY_STEP_PX` |
| Ray length | $L(\theta)$ = centroid → droplet wall | px | **[code]** `ray_length_px` |
| **Normalised radius** | $\rho = d / L$ | — | **[code]** `Rho_Normalized`. **Measured from the CENTROID**, normalised by ray length. |
| Wall proximity index | $W = 1 - \rho$ | — | **[code]** `W_Wall_Proximity_Index` |
| **Ray start (nuclear edge)** | $\rho_0(\theta) = \min_i \rho_i = \dfrac{r_{\text{nuc}}(\theta)}{L(\theta)}$ | — | Because the interior is excluded, the smallest $\rho$ on a ray **is** the nuclear boundary. Doubles as a QC statistic (§9). |
| **Edge-normalised radius** | $\rho_e = \dfrac{\rho - \rho_0}{1 - \rho_0}$ | — | **The correction.** $0$ = nuclear boundary, $1$ = droplet wall. A fixed cut on $\rho$ instead of $\rho_e$ selects a region inside the excluded nucleus and silently discards most rays. |
| Ray length (physical) | $L_{\mu m} = r_{\text{nuc},\mu m} / \rho_0$ | µm | Recovers physical units without `distance_px` in the export |
| Distance from centroid | $r_{\mu m} = \rho \, L_{\mu m}$ | µm | |
| Distance from nuclear edge | $e_{\mu m} = (\rho - \rho_0)\, L_{\mu m}$ | µm | The right axis for depth bands |
| Sector assignment | $\mathrm{sec}(\theta) = \lfloor \theta / \Delta_{\text{sec}} \rfloor \Delta_{\text{sec}}$ | deg | $\Delta_{\text{sec}} = 360/n_{\text{sec}}$ |
| Sector mean intensity | $\bar I_{\text{sec}} = \mathrm{mean}\{I : \theta \in \mathrm{sec},\ \rho_e \le \rho_{\text{peri}}\}$ | — | **Mean, not sum** — summing lets longer rays contribute more points and reintroduces the geometric bias |
| Relative sector intensity | $\tilde I = \bar I_{\text{sec}} / \langle \bar I_{\text{sec}} \rangle_{\text{panel}}$ | — | Removes the timepoint baseline so asymmetry is what is plotted |

---

## 8. Radial features and circular statistics

| Quantity | Formula | Notes |
|---|---|---|
| Smoothed profile | $\tilde I_i = \dfrac{1}{2k+1}\sum_{m=-k}^{k} I_{i+m}$ | Boxcar, half-width $k$ |
| Peak (fold) criterion | $\tilde I_i \ge \max_{|m|\le k} \tilde I_{i+m}$ **and** $\tilde I_i - \min_{|m|\le k}\tilde I_{i+m} > \alpha\,(\max \tilde I - \min \tilde I)$ | $\alpha$ = `min_rel`. The prominence term is what excludes shot noise. |
| Fold count | $F(\theta)$ = number of qualifying peaks on that ray | Trim the outer band first, or the oil interface counts as a fold |
| Envelope peak radius | $r_{\text{env}}(\theta) = d\big(\arg\max_i I_i\big)\, s$ | Threshold-free nuclear radius estimate. Median across rays gave 7.0 µm vs 6.2 µm from the mask. |
| **Resultant vector** | $C = \dfrac{\sum_j w_j \cos\theta_j}{\sum_j w_j}$, $\;S = \dfrac{\sum_j w_j \sin\theta_j}{\sum_j w_j}$ | $w_j$ = intensity weight |
| Resultant length | $R = \sqrt{C^2 + S^2}$ | $0$ = uniform, $1$ = fully polarised. **Near 1 whenever angular coverage is narrow** — always report alongside $n_\theta$. |
| Mean direction | $\bar\theta = \mathrm{atan2}(S, C) \bmod 360$ | deg |
| Circular variance | $V = 1 - R$ | |
| Rayleigh test | $Z = n R^2$, $\;p \approx e^{-Z}\left(1 + \dfrac{2Z - Z^2}{4n}\right)$ | Tests uniformity. Valid only for genuinely independent directions. |
| Coxcomb radius | $r_{\text{plot}} = \sqrt{v}$ | Makes wedge **area** ∝ $v$. Plain radius makes area ∝ $v^2$, exaggerating differences — state which is used in any figure legend. |

---

## 9. Geometric QC

The core constraint: a nucleus of radius $r$ inside a droplet of radius $R$ can
only intersect planes within $\pm r$ of the droplet's equatorial plane.

| Quantity | Formula | Notes |
|---|---|---|
| Droplet equatorial plane | $z_{\text{eq}} = \arg\max_z A_{\text{drop}}(z)$ | Per droplet, per timepoint |
| Droplet radius | $R = \sqrt{A_{\text{drop}}(z_{\text{eq}})/\pi}$ | Equatorial value only |
| **Cap index** | $u = \dfrac{\lvert z - z_{\text{eq}}\rvert \, \Delta z}{R}$ | $0$ = equator, $1$ = pole. $u > r/R \approx 0.53$ ⇒ **no nucleus can exist in that plane** of that droplet. Used at $u > 0.75$ to mine guaranteed-negative training patches. |
| Predicted $\rho_0$ vs depth | $\rho_0(d) = \dfrac{\sqrt{r^2 - d^2}}{\sqrt{R^2 - d^2}}$ | Maximised at the equator, where it equals $r/R$ |
| $\rho_0$ validity bound | real nucleus ⇒ $\rho_0 \lesssim r/R$ | **Caveat:** measured $\rho_0$ is $\min$ per ray, so off-centre nuclei inflate it. Observed late-time median 0.59 with an upper quartile of 0.67 — treat $\rho_0$ as a QC statistic, **not** a standalone filter. |
| Scale-free nuclear scaling | $\rho_0^2 = A_{\text{nuc}}/A_{\text{drop}}$ | **Independent of $s$.** Observed 0.42 → 0.59 from t=3→9, i.e. area ratio 0.18 → 0.35. |
| Lab-frame pooled asymmetry | $R_{\text{pooled}}$ over all nuclei without rotation | Should be ≈ 0 if asymmetry is biological and randomly oriented. A preferred direction indicates an imaging artefact. |
| False-positive rate | $\mathrm{FPR}(t) = \dfrac{1}{n}\sum \mathbb{1}[\rho_0 > \rho_{\max}]$ | Per-timepoint QC metric; the target for v10 retraining |

---

## 10. Model fits and filters

| Quantity | Formula | Notes |
|---|---|---|
| Saturating radius growth | $r(t) = r_0 + (r_{\max} - r_0)\left(1 - e^{-(t - t_0)/\tau}\right)$ | $t_0$ = onset lag (~7 min in the hand-calculated data) |
| Predicted area | $A(t) = \pi\, r(t)^2$ | Fit via `nls` on $A$, not $r$, to weight by the observable |
| Logistic import | $f_{NC}(t) = 0.5 + \dfrac{P - 0.5}{1 + e^{-(t - t_{50})/k}}$ | $P$ = plateau (~0.9), $t_{50}$ = half-maximal time. Floor is 0.5, not 0. |
| Size envelope (multiplicative) | $A_{\min}(t) = \lambda_{\text{lo}} A_{\text{exp}}(t)$, $\;A_{\max}(t) = \lambda_{\text{hi}} A_{\text{exp}}(t)$ | Multiplicative because spread scales with size |
| Size envelope (anchored) | linear interpolation between $(t_k, A_k)$ anchors, held flat outside the range | Closer to how domain expertise is usually stated |
| Band width | $\lambda_{\text{hi}} / \lambda_{\text{lo}}$ | **Report this.** A narrow band tracking the expected curve manufactures the growth it is meant to measure. |
| Classification metrics | $\mathrm{prec} = \tfrac{TP}{TP+FP}$, $\mathrm{rec} = \tfrac{TP}{TP+FN}$, $F_1 = \tfrac{2\,\mathrm{prec}\,\mathrm{rec}}{\mathrm{prec}+\mathrm{rec}}$ | Requires per-object ground truth |
| Aggregate fit error | $\mathrm{RMSE} = \sqrt{\dfrac1n\sum (\hat A_i - A_i^{\text{true}})^2}$ | Weaker than per-object labels: matching a median does not prove the right objects were removed |

---

## Quantities that are calibration-independent

Worth knowing while $s$ remains unresolved — these hold whatever the stage
micrometer says, because they are ratios:

- N/C ratio and N/C fraction
- Nucleus/droplet area fraction $f$, and $r/R = \sqrt f$
- $\rho_0$, $\rho$, $\rho_e$, and everything derived from them
- Resultant length $R$ and mean direction $\bar\theta$
- Fold counts
- Cap index $u$
- Solidity, circularity, eccentricity

Everything expressed in µm, µm², or µm³ scales with $s$ or $s^2$ or $s^3$ and
must be recomputed once the calibration is settled.
