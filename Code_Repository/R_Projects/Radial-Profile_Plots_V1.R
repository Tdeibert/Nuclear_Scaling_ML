library(tidyverse)
library(ggplot2)



#### Data Frame Creation #### 
run <-
nuclei <- read.csv("~/Projects/Nuclear_Scaling/Data/Fake_Data/Nuclei.csv")
zstack <- read.csv("~/Projects/Nuclear_Scaling/Data/Fake_Data/NucleusZStack.csv")
radial <- read.csv("~/Projects/Nuclear_Scaling/Data/Fake_Data/RadialProfile.csv")
raw <- read.csv("~/Projects/Nuclear_Scaling/Data/Fake_Data/RawIntensities.csv")












# =============================================================================
# radial_profile_plots.R
#
# Brainstorm set: every radial-profile visualisation we discussed, in one place,
# so they can be run against real data and compared.
#
#   A  raster heatmap            theta x rho, faceted by timepoint
#   B  draped wireframe surface  the 2.5D perspective plot (lattice)
#   C  rose diagram              linear radius = mean intensity per sector
#   D  coxcomb / Nightingale     sqrt radius, so wedge AREA is proportional
#   E  positioned wind rose      radius = physical distance, fill = intensity
#   F  stacked wind rose         radius = accumulated intensity, fill = depth band
#   G  ray-length rose           wedge length = distance to droplet edge
#   H  circular statistics       resultant vector R and direction vs time
#   I  pooled population rose    QC: is any asymmetry in the lab frame?
#
# Each block is standalone once Section 0 has run. Untested against real data —
# column names are taken from the export schema, so expect minor fixes.
# =============================================================================

library(tidyverse)

# -----------------------------------------------------------------------------
# 0. Setup: paths, load, and capability check
# -----------------------------------------------------------------------------

RUN_DIR <- "~/Projects/Nuclear_Scaling/Data/Fake_Data/"   # <-- set this
EXPORTS <- file.path(RUN_DIR, "exports")

# Pixel size from the run's own config snapshot, so figures inherit whatever
# calibration that run used rather than a value hardcoded here. Falls back if
# jsonlite isn't installed.
PX <- tryCatch(
  jsonlite::fromJSON(file.path(RUN_DIR, "config.json"))$pixel_size_um,
  error = function(e) { warning("config.json unreadable; using 0.2167"); 0.2167 }
)
message("pixel_size_um = ", PX)

radial <- read_csv(file.path(EXPORTS, "RadialProfile.csv"), show_col_types = FALSE)
nuclei <- read_csv(file.path(EXPORTS, "Nuclei.csv"),        show_col_types = FALSE)

# The export currently ships Rho_Normalized but may not ship absolute distance.
# Plots E/F/G need physical radius; this flag lets them degrade gracefully.
HAS_ABS <- all(c("distance_px", "ray_length_px") %in% names(radial))
message("absolute distance available: ", HAS_ABS)
if (!HAS_ABS) message("  -> E/F/G will use normalised radius; add distance_px ",
                      "and ray_length_px to the RadialProfile export for physical units.")

CHANNEL <- "Membrane"
NUC     <- radial |>                      # nucleus present at the most timepoints
  filter(Channel == CHANNEL) |>
  count(Nucleus_ID, Time_Frame) |>
  count(Nucleus_ID, name = "n_t") |>
  slice_max(n_t, n = 1) |>
  pull(Nucleus_ID) |>
  first()
message("using Nucleus_ID = ", NUC)

N_SECTORS <- 18                            # 18 -> 20 deg; 6 -> 60 deg
SEC_W     <- 360 / N_SECTORS

# Optional: subtract per-nucleus background before any intensity comparison.
# RawIntensities carries Background_Membrane / _NPC / _Mcherry.
SUBTRACT_BG <- FALSE
if (SUBTRACT_BG) {
  bg <- read_csv(file.path(EXPORTS, "RawIntensities.csv"), show_col_types = FALSE) |>
    select(Nucleus_ID, Time_Frame, bg = Background_Membrane)
  radial <- radial |>
    left_join(bg, by = c("Nucleus_ID", "Time_Frame")) |>
    mutate(Intensity = pmax(Intensity - coalesce(bg, 0), 0))
}

# Common prep: one nucleus, one channel, with sector and radius columns added.
prep <- function(df, nucleus_id = NUC, channel = CHANNEL) {
  out <- df |>
    filter(Nucleus_ID == nucleus_id, Channel == channel) |>
    mutate(sector     = floor(Theta_deg / SEC_W) * SEC_W,
           sector_mid = sector + SEC_W / 2,
           theta_rad  = Theta_deg * pi / 180)
  if (HAS_ABS) out <- out |> mutate(r_um = distance_px * PX)
  else         out <- out |> mutate(r_um = Rho_Normalized)   # unitless fallback
  out
}

dat <- prep(radial)
R_LAB <- if (HAS_ABS) "distance from nucleus (\u00b5m)" else "normalised radius"

# -----------------------------------------------------------------------------
# A. Raster heatmap — theta x rho, faceted by timepoint
#    The plain look at the data. Vertical band near r=0 is the envelope,
#    band near r=1 is the oil interface.
# -----------------------------------------------------------------------------

pA <- dat |>
  mutate(rbin = cut(Rho_Normalized, breaks = seq(0, 1, length.out = 61),
                    labels = FALSE, include.lowest = TRUE) / 60) |>
  group_by(Time_Frame, rbin, Theta_deg) |>
  summarise(Intensity = mean(Intensity), .groups = "drop") |>
  ggplot(aes(rbin, Theta_deg, fill = Intensity)) +
  geom_raster(interpolate = TRUE) +
  facet_wrap(~ Time_Frame) +
  scale_fill_viridis_c(option = "inferno") +
  scale_y_continuous(breaks = seq(0, 360, 90)) +
  labs(x = "r (nucleus edge \u2192 droplet edge)", y = "\u03b8 (deg)",
       title = paste0("Radial membrane profile \u2014 nucleus ", NUC)) +
  theme_minimal(base_size = 9)
print(pA)

# -----------------------------------------------------------------------------
# B. Draped wireframe surface (lattice) — the perspective plot
#    Same grid as A, with intensity lifted into z instead of encoded as colour.
#    Needs a COMPLETE grid, hence the expand/join.
# -----------------------------------------------------------------------------

if (requireNamespace("lattice", quietly = TRUE)) {
  TF <- max(dat$Time_Frame)                       # one timepoint per surface
  
  surf <- dat |>
    filter(Time_Frame == TF) |>
    mutate(rho   = round(Rho_Normalized * 40) / 40,
           theta = floor(Theta_deg / 6) * 6) |>
    group_by(rho, theta) |>
    summarise(Intensity = mean(Intensity), .groups = "drop")
  
  grid_full <- expand_grid(rho   = seq(0, 1, by = 1/40),
                           theta = seq(0, 354, by = 6)) |>
    left_join(surf, by = c("rho", "theta")) |>
    group_by(rho) |>
    mutate(Intensity = ifelse(is.na(Intensity), mean(Intensity, na.rm = TRUE), Intensity)) |>
    ungroup() |>
    mutate(Intensity = ifelse(is.finite(Intensity), Intensity, 0))
  
  print(lattice::wireframe(
    Intensity ~ rho * theta, data = grid_full,
    drape = TRUE, colorkey = TRUE, shade = FALSE,
    col.regions = hcl.colors(100, "Inferno"),
    screen = list(z = 30, x = -60), aspect = c(1, 0.6),
    xlab = "r (norm)", ylab = list("\u03b8 (deg)", rot = 30), zlab = "",
    main = paste0("Nucleus ", NUC, " \u2014 t=", TF)
  ))
} else message("skip B: install.packages('lattice')")

# -----------------------------------------------------------------------------
# C. Rose diagram — linear radius = mean intensity in a perinuclear band
#    Simplest version. NOTE: wedge area scales as value^2, which visually
#    exaggerates differences. Use D for anything going into a figure.
# -----------------------------------------------------------------------------

PERI <- 0.30        # inner fraction of the ray treated as perinuclear

roseC <- dat |>
  filter(Rho_Normalized <= PERI) |>
  group_by(Time_Frame, sector, sector_mid) |>
  summarise(mean_int = mean(Intensity), n = n(), .groups = "drop")

pC <- ggplot(roseC, aes(sector_mid, mean_int, fill = mean_int)) +
  geom_col(width = SEC_W, colour = "grey25", linewidth = .15) +
  coord_polar(start = 0) +
  scale_x_continuous(breaks = seq(0, 360 - 60, 60), limits = c(0, 360)) +
  scale_fill_viridis_c(option = "inferno", guide = "none") +
  facet_wrap(~ Time_Frame) +
  labs(x = NULL, y = "mean membrane intensity",
       title = paste0("Rose \u2014 nucleus ", NUC, ", r \u2264 ", PERI, " of ray")) +
  theme_minimal(base_size = 9)
print(pC)

# -----------------------------------------------------------------------------
# D. Coxcomb / Nightingale — sqrt radius so wedge AREA is proportional to value.
#    This is the honest version for publication; state the scaling in the legend.
# -----------------------------------------------------------------------------

pD <- roseC |>
  mutate(r_plot = sqrt(mean_int)) |>
  ggplot(aes(sector_mid, r_plot, fill = mean_int)) +
  geom_col(width = SEC_W, colour = "white", linewidth = .2) +
  coord_polar(start = 0) +
  scale_x_continuous(breaks = seq(0, 360 - 60, 60), limits = c(0, 360)) +
  scale_y_continuous(labels = function(x) round(x^2)) +   # ticks back in real units
  scale_fill_viridis_c(option = "inferno", name = "mean\nintensity") +
  facet_wrap(~ Time_Frame) +
  labs(x = NULL, y = "mean intensity (area-proportional radius)",
       title = paste0("Coxcomb \u2014 nucleus ", NUC)) +
  theme_minimal(base_size = 9)
print(pD)

# -----------------------------------------------------------------------------
# E. Positioned wind rose — radius = physical distance, fill = intensity.
#    Each wedge ends where its rays end, so the outline traces the droplet
#    boundary as seen from the nucleus. Asymmetric outline = nucleus off-centre.
# -----------------------------------------------------------------------------

RBIN <- if (HAS_ABS) 1.5 else 0.05     # µm, or fraction of ray

windE <- dat |>
  mutate(rbin = floor(r_um / RBIN) * RBIN) |>
  group_by(Time_Frame, sector, rbin) |>
  summarise(Intensity = mean(Intensity), .groups = "drop")

pE <- ggplot(windE) +
  geom_rect(aes(xmin = sector, xmax = sector + SEC_W,
                ymin = rbin,   ymax = rbin + RBIN,
                fill = Intensity), colour = NA) +
  coord_polar(start = 0) +
  scale_x_continuous(breaks = seq(0, 360 - 60, 60), limits = c(0, 360)) +
  scale_fill_viridis_c(option = "inferno") +
  facet_wrap(~ Time_Frame) +
  labs(x = NULL, y = R_LAB,
       title = paste0("Positioned wind rose \u2014 nucleus ", NUC)) +
  theme_minimal(base_size = 9)
print(pE)

# Variant: cap the radial axis to drop the oil interface, which is bright in
# every sector and otherwise dominates the colour scale.
if (HAS_ABS) {
  print(pE + coord_polar(start = 0) +
          scale_y_continuous(limits = c(0, 8)) +
          labs(subtitle = "perinuclear only (0\u20138 \u00b5m)"))
}

# -----------------------------------------------------------------------------
# F. Stacked wind rose — radius = accumulated intensity, fill = depth band.
#    Wedge length is total signal in the sector; stacking shows which depth
#    it came from. MEAN per band, not sum, so long rays don't inflate.
# -----------------------------------------------------------------------------

# Bands must fit inside the SHORTEST ray or they silently go missing.
ray_check <- dat |>
  group_by(Time_Frame, Theta_deg) |>
  summarise(len = max(r_um), .groups = "drop") |>
  summarise(min_ray = min(len), median_ray = median(len))
print(ray_check)

BREAKS <- if (HAS_ABS) c(0, 2, 4, 6, 8) else c(0, .1, .2, .3, .4)
LABS   <- paste0(head(BREAKS, -1), "\u2013", tail(BREAKS, -1))

stk <- dat |>
  mutate(band = cut(r_um, breaks = BREAKS, labels = LABS, include.lowest = TRUE)) |>
  filter(!is.na(band)) |>
  group_by(Time_Frame, sector, sector_mid, band) |>
  summarise(mean_int = mean(Intensity), .groups = "drop")

pF <- ggplot(stk, aes(sector_mid, mean_int, fill = band)) +
  geom_col(width = SEC_W, colour = "white", linewidth = .12, position = "stack") +
  coord_polar(start = 0) +
  scale_x_continuous(breaks = seq(0, 360 - 60, 60), limits = c(0, 360)) +
  scale_fill_viridis_d(option = "mako", direction = -1,
                       name = if (HAS_ABS) "distance (\u00b5m)" else "radius (norm)") +
  facet_wrap(~ Time_Frame) +
  labs(x = NULL, y = "stacked mean intensity",
       title = paste0("Stacked wind rose \u2014 nucleus ", NUC)) +
  theme_minimal(base_size = 9)
print(pF)

# -----------------------------------------------------------------------------
# G. Ray-length rose — wedge length = distance from nucleus to droplet edge.
#    Pure geometry, no intensity. Asymmetry here means the nucleus sits
#    off-centre in its droplet, which is an independent readout of aster
#    pushing and is not currently extracted anywhere in the pipeline.
# -----------------------------------------------------------------------------

geo <- dat |>
  group_by(Time_Frame, sector, sector_mid, Theta_deg) |>
  summarise(ray = max(r_um), .groups = "drop") |>
  group_by(Time_Frame, sector, sector_mid) |>
  summarise(ray = mean(ray), .groups = "drop")

pG <- ggplot(geo, aes(sector_mid, ray, fill = ray)) +
  geom_col(width = SEC_W, colour = "grey30", linewidth = .15) +
  coord_polar(start = 0) +
  scale_x_continuous(breaks = seq(0, 360 - 60, 60), limits = c(0, 360)) +
  scale_fill_viridis_c(option = "cividis", guide = "none") +
  facet_wrap(~ Time_Frame) +
  labs(x = NULL, y = R_LAB,
       title = paste0("Nucleus\u2013to\u2013droplet distance by direction \u2014 nucleus ", NUC),
       subtitle = "asymmetry = nucleus displaced from droplet centre") +
  theme_minimal(base_size = 9)
print(pG)

# -----------------------------------------------------------------------------
# H. Circular statistics — the quantitative complement to the roses.
#    Intensity-weighted resultant vector: R near 0 = uniform, near 1 = polarised.
#    Computed in base R so no extra package is needed; `circular` only if you
#    want the Rayleigh test.
# -----------------------------------------------------------------------------

resultant <- function(theta_rad, w) {
  w <- pmax(w, 0)
  if (sum(w) == 0) return(tibble(R = NA_real_, dir_deg = NA_real_))
  C <- sum(w * cos(theta_rad)) / sum(w)
  S <- sum(w * sin(theta_rad)) / sum(w)
  tibble(R = sqrt(C^2 + S^2),
         dir_deg = (atan2(S, C) * 180 / pi) %% 360)
}

circ <- dat |>
  filter(Rho_Normalized <= PERI) |>
  group_by(Time_Frame) |>
  group_modify(~ resultant(.x$theta_rad, .x$Intensity)) |>
  ungroup()
print(circ)

pH <- ggplot(circ, aes(Time_Frame, R)) +
  geom_line(linewidth = .8) + geom_point(aes(colour = dir_deg), size = 3) +
  scale_colour_gradientn(colours = hcl.colors(12, "Spectral"),
                         limits = c(0, 360), name = "direction\n(deg)") +
  labs(x = "Time frame", y = "resultant length R (asymmetry)",
       title = paste0("Membrane asymmetry over time \u2014 nucleus ", NUC),
       subtitle = "R = 0 uniform, R = 1 fully polarised") +
  theme_minimal(base_size = 10)
print(pH)

# Same, across all nuclei — the version worth actually testing.
circ_all <- radial |>
  filter(Channel == CHANNEL, Rho_Normalized <= PERI) |>
  mutate(theta_rad = Theta_deg * pi / 180) |>
  group_by(Nucleus_ID, Time_Frame) |>
  group_modify(~ resultant(.x$theta_rad, .x$Intensity)) |>
  ungroup()

pH2 <- ggplot(circ_all, aes(factor(Time_Frame), R)) +
  geom_boxplot(outlier.size = .5, fill = "grey90") +
  labs(x = "Time frame", y = "resultant length R",
       title = "Membrane asymmetry across all nuclei") +
  theme_minimal(base_size = 10)
print(pH2)

# -----------------------------------------------------------------------------
# I. Pooled population rose — QC, not biology.
#    Pooling every nucleus in the LAB frame: if real asymmetry is biological and
#    randomly oriented, this should be flat. A consistent direction here points
#    at an imaging artefact (uneven illumination, chromatic shift, stitching).
#    Run this before believing any per-nucleus asymmetry.
# -----------------------------------------------------------------------------

pI <- radial |>
  filter(Channel == CHANNEL, Rho_Normalized <= PERI) |>
  mutate(sector = floor(Theta_deg / SEC_W) * SEC_W + SEC_W / 2) |>
  group_by(Time_Frame, sector) |>
  summarise(mean_int = mean(Intensity), .groups = "drop") |>
  ggplot(aes(sector, mean_int, fill = mean_int)) +
  geom_col(width = SEC_W, colour = "white", linewidth = .12) +
  coord_polar(start = 0) +
  scale_x_continuous(breaks = seq(0, 360 - 60, 60), limits = c(0, 360)) +
  scale_fill_viridis_c(option = "inferno", guide = "none") +
  facet_wrap(~ Time_Frame) +
  labs(x = NULL, y = "mean intensity",
       title = "Pooled rose, all nuclei (lab frame)",
       subtitle = "should be flat \u2014 a preferred direction indicates an artefact") +
  theme_minimal(base_size = 9)
print(pI)

# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
# fig_dir <- file.path(RUN_DIR, "figures"); dir.create(fig_dir, showWarnings = FALSE)
# for (nm in c("pA","pC","pD","pE","pF","pG","pH","pH2","pI"))
#   ggsave(file.path(fig_dir, paste0(nm, ".png")), get(nm),
#          width = 9, height = 7, dpi = 300)
