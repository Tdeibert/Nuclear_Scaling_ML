# ===========================================================
# 0) PACKAGES
# ===========================================================
library(terra)
library(tidyverse)   # dplyr, tidyr, purrr, tibble, stringr, ggplot2, readr
library(glue)
library(broom)
library(scales)
library(ggnewscale)
# ===========================================================
# 1) USER SETTINGS (EDIT THESE)
# ===========================================================
# Multi-band stack with channels × Z × time (same for all channels)
stack_path      <- "C:/Users/tdeibert/.Working_Docs_Folder/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 2/Region 4/Region_4_Droplet_T10.tif"   # <-- set this
n_channels      <- 3
n_z             <- 20
n_time          <- 1
channel_names   <- c("mCherry","GFP","Alexaflour647")       # <-- set your names
pixel_size_um   <- 0.108                      # μm per pixel

# Which is nuclear (NLS) and which is membrane channel?
nuclear_channel <- 2
membrane_channel<- 1   # set to your membrane channel index (1..n_channels)

# Output
out_dir         <- "nucleus_mask_and_polar_outputs"
dir.create(out_dir, showWarnings = FALSE)

# ------------------ Masking parameters (robust defaults) ------------------
# Gaussian blur before thresholding (in pixels)
gauss_sigma_px      <- 1.5     # smoothing strength
gauss_radius_px     <- ceiling(gauss_sigma_px*3)  # kernel half-size

# Otsu threshold multiplier (>=1 tightens mask to brightest core; <1 expands)
otsu_multiplier     <- 1.00

# Remove tiny specks below this area (μm^2)
min_obj_area_um2    <- 25
min_obj_area_px     <- ceiling(min_obj_area_um2 / (pixel_size_um^2))

# Keep only the largest connected component? (common for single nucleus per FOV)
keep_largest        <- TRUE

# Fill interior holes (recommended)
fill_holes          <- TRUE

# Save intermediate mask QC PNGs for t == 1 only (to avoid tons of files)
save_mask_qc_png    <- TRUE
save_only_t1        <- TRUE

# ------------------ Polar analysis parameters ------------------
n_theta             <- 360     # angular samples (1 deg)
radial_step_px      <- 1       # radial sampling step in px
min_radius_px       <- 2
max_radius_mode     <- "edge"  # or set max_radius_px <- 120

# Save polar-derived plots (for t == 1 to limit I/O)
save_kymograph_png  <- TRUE
save_mode_heat_png  <- TRUE
save_radial_png     <- TRUE
save_dominant_png   <- TRUE
save_rose_png       <- TRUE
rose_radius_px      <- 30
save_tables_csv     <- TRUE

# ===========================================================
# 2) LOAD STACK & NAME LAYERS
# ===========================================================
r <- rast(stack_path)
stopifnot(nlyr(r) == n_channels * n_z * n_time)

layer_index <- tidyr::expand_grid(
  ch = seq_len(n_channels),
  z  = seq_len(n_z),
  t  = seq_len(n_time)
) %>% mutate(name = glue("ch{ch}_z{z}_t{t}"))
names(r) <- layer_index$name

# Convenience helpers to index layers
layer_name <- function(ch, z, t) glue("ch{ch}_z{z}_t{t}")
idx_of     <- function(ch, z, t) which(names(r) == layer_name(ch,z,t))

# ===========================================================
# 3) IMAGE + MATH HELPERS
# ===========================================================
# Build a normalized 2D Gaussian kernel for terra::focal
gaussian_kernel <- function(sigma, radius) {
  xs <- -radius:radius
  g1 <- exp(-(xs^2) / (2*sigma^2))
  ker <- outer(g1, g1, "*")
  ker / sum(ker)
}

# Otsu threshold (simple, fast)
otsu_threshold <- function(v) {
  v <- v[is.finite(v)]
  v <- v[v >= quantile(v, 0.001) & v <= quantile(v, 0.999)] # trim huge outliers
  if (length(v) < 256) return(mean(v, na.rm=TRUE))
  h <- hist(v, breaks = 256, plot = FALSE)
  p <- h$counts / sum(h$counts)
  omega <- cumsum(p)
  mu <- cumsum(p * h$mids)
  mu_t <- mu[length(mu)]
  sigma_b2 <- (mu_t * omega - mu)^2 / (omega * (1 - omega) + 1e-12)
  thr <- h$mids[which.max(replace(sigma_b2, !is.finite(sigma_b2), -Inf))]
  thr
}

# Fill holes by polygonizing and dissolving, then rasterizing back
fill_binary_holes <- function(bin_r) {
  # bin_r: SpatRaster with 0/1 values
  # Convert to polygons and dissolve holes
  p <- as.polygons(bin_r, dissolve = TRUE, values = TRUE, trunc = TRUE)
  if (nrow(p) == 0) return(bin_r)
  # Rasterize back to original grid
  r_out <- rast(bin_r)
  rp <- rasterize(p, r_out, field = 1, background = 0, touches = TRUE)
  rp
}

# ===========================================================
# 4) NUCLEAR MASK DERIVATION FROM CHANNEL 1 (NLS)
# ===========================================================
# Prebuild Gaussian kernel
gker <- gaussian_kernel(gauss_sigma_px, gauss_radius_px)

# Function to derive mask for a single NLS layer (ch == nuclear_channel)
derive_nuclear_mask <- function(nls_layer_r) {
  # Smooth
  sm  <- focal(nls_layer_r, w = gker, fun = "sum", na.policy = "omit", na.rm = TRUE)
  # Normalize sm to mean=0, sd=1 (robust z-score)
  v <- values(sm, mat=FALSE)
  m <- median(v, na.rm=TRUE); s <- mad(v, constant = 1, na.rm=TRUE) + 1e-9
  zn <- (sm - m) / s
  
  # Convert to matrix for thresholding
  zmat <- as.matrix(zn, wide = TRUE)
  thr  <- otsu_threshold(zmat)*otsu_multiplier
  bin  <- zmat > thr
  
  # Remove tiny components via clumping
  bin_r <- rast(bin); ext(bin_r) <- ext(nls_layer_r); crs(bin_r) <- crs(nls_layer_r)
  # Connected components (8-neighborhood)
  cl <- patches(bin_r, directions = 8, zeroAsNA = TRUE)
  if (!is.null(min_obj_area_px) && is.finite(min_obj_area_px) && min_obj_area_px > 1) {
    ff <- freq(cl) %>% 
      as_tibble() %>% 
      rename(label = value, npx = count)
    keep <- ff %>% filter(npx >= min_obj_area_px) %>% pull(label)
    cl <- classify(cl, rcl = cbind(setdiff(unique(values(cl)), keep), NA))
  }
  
  # Keep largest component (optional)
  if (keep_largest) {
    ff <- freq(cl) %>% as_tibble() %>% rename(label = value, npx = count)
    if (nrow(ff) > 0) {
      lab <- ff$label[which.max(ff$npx)]
      cl <- classify(cl, rcl = cbind(setdiff(unique(values(cl)), lab), NA))
    }
  }
  
  mask <- clamp(!is.na(cl), lower=0, upper=1, values=TRUE)
  if (fill_holes) {
    mask <- fill_binary_holes(mask)
  }
  mask
}

# Build a full mask stack aligned with r (mask exists only for ch==nuclear_channel; others just copied for alignment)
message("Deriving nuclear masks from channel: ", nuclear_channel)
mask_stack <- rast()
for (z in seq_len(n_z)) {
  for (t in seq_len(n_time)) {
    # NLS layer for this z,t
    idx_nls <- idx_of(nuclear_channel, z, t)
    nls_r   <- r[[idx_nls]]
    # derive mask
    mk      <- derive_nuclear_mask(nls_r)
    names(mk) <- names(nls_r)
    # append to mask_stack with same number of channels? We’ll store per (z,t) once and reuse.
    mask_stack <- c(mask_stack, mk)
  }
}
# mask_stack has n_z * n_time layers, named ch{nuclear_channel}_z{z}_t{t}

# Quick QC overlay plots for t==1 (optional)
if (save_mask_qc_png) {
  qc_dir <- file.path(out_dir, "mask_qc")
  dir.create(qc_dir, showWarnings = FALSE)
  for (z in seq_len(n_z)) {
    t <- 1
    nm <- layer_name(nuclear_channel, z, t)
    nls_r <- r[[idx_of(nuclear_channel, z, t)]]
    mk    <- mask_stack[[which(names(mask_stack) == nm)]]
    # build a quick gg image
    im  <- as.matrix(nls_r, wide = TRUE)
    mkm <- as.matrix(mk, wide = TRUE)
    df  <- tibble(
      x = rep(1:ncol(im), each = nrow(im)),
      y = rep(1:nrow(im), times = ncol(im)),
      I = as.vector(im),
      M = as.vector(mkm)
    )
    p <- ggplot(df, aes(x, y)) +
      geom_raster(aes(fill = I)) +
      scale_fill_viridis_c(option = "magma") +
      coord_equal() + theme_void() +
      new_scale("fill") +
      geom_raster(aes(fill = factor(M)), alpha = 0.35) +
      scale_fill_manual(values = c("0" = NA, "1" = "cyan")) +
      ggtitle(glue("Mask QC: {nm}"))
    ggsave(file.path(qc_dir, glue("{nm}_mask_qc.png")), p, width = 5.5, height = 5, dpi = 300)
  }
}

# ===========================================================
# 5) POLAR FOURIER ANALYSIS (NUCLEUS-CENTRIC) ON MEMBRANE CHANNEL
# ===========================================================
# Helpers for polar analysis
centroid_from_mask <- function(mask_mat) {
  m <- mask_mat; m[is.na(m)] <- 0
  w <- sum(m); if (w <= 0) return(c(NA_real_, NA_real_))
  nr <- nrow(m); nc <- ncol(m)
  xs <- matrix(rep(1:nc, each = nr), nrow = nr)
  ys <- matrix(rep(1:nr, nc), nrow = nr)
  c(sum(xs*m)/w, sum(ys*m)/w)
}
bilinear_sample <- function(img, x, y) {
  nr <- nrow(img); nc <- ncol(img)
  x <- pmin(pmax(x, 1), nc - 1); y <- pmin(pmax(y, 1), nr - 1)
  x0 <- floor(x); x1 <- x0 + 1; y0 <- floor(y); y1 <- y0 + 1
  dx <- x - x0; dy <- y - y0
  i00 <- img[cbind(y0, x0)]; i10 <- img[cbind(y0, x1)]
  i01 <- img[cbind(y1, x0)]; i11 <- img[cbind(y1, x1)]
  i0 <- i00*(1-dx) + i10*dx; i1 <- i01*(1-dx) + i11*dx
  i0*(1-dy) + i1*dy
}
sample_circle_intensity <- function(img, cx, cy, r, n_theta = 360) {
  thetas <- seq(0, 2*pi, length.out = n_theta + 1)[- (n_theta + 1)]
  xs <- cx + r*cos(thetas); ys <- cy + r*sin(thetas)
  tibble(theta = thetas, I = bilinear_sample(img, xs, ys))
}
angular_fft <- function(I_theta) {
  n <- nrow(I_theta)
  F <- fft(I_theta$I)
  tibble(m = 0:(n-1), power = (Mod(F)^2)/n) %>% slice(1:(floor(n/2)+1))
}
max_radius_to_edge <- function(nr, nc, cx, cy) {
  floor(min(cx-1, nc-cx, cy-1, nr-cy))
}
make_kymograph <- function(img, cx, cy, r_seq, n_theta) {
  map_dfr(r_seq, ~ mutate(sample_circle_intensity(img, cx, cy, .x, n_theta), r = .x))
}
make_mode_radius <- function(img, cx, cy, r_seq, n_theta) {
  map_dfr(r_seq, ~ mutate(angular_fft(sample_circle_intensity(img, cx, cy, .x, n_theta)), r = .x))
}
dominant_mode_summary <- function(mode_r_df, m_exclude0 = TRUE) {
  df <- if (m_exclude0) filter(mode_r_df, m > 0) else mode_r_df
  df %>% group_by(r) %>%
    summarize(dom_m   = m[which.max(power)],
              dom_pow = max(power),
              sum_pow = sum(power),
              ani_idx = ifelse(sum_pow > 0, dom_pow/sum_pow, NA_real_), .groups = "drop")
}

# Plotters
plot_kymograph <- function(kymo_df) {
  kymo_df %>%
    mutate(theta_deg = theta*180/pi, r_um = r*pixel_size_um) %>%
    ggplot(aes(theta_deg, r_um, fill = I)) +
    geom_raster() + scale_y_reverse() +
    labs(title = "θ–r Intensity (Kymograph)", x = expression(theta~"(deg)"), y = "Radius (µm)", fill = "I") +
    theme_bw()
}
plot_mode_heatmap <- function(mode_r_df) {
  mode_r_df %>%
    mutate(r_um = r*pixel_size_um) %>%
    ggplot(aes(m, r_um, fill = power)) +
    geom_raster() + labs(title = "Angular Fourier Power vs Radius", x = "Mode (m)", y = "Radius (µm)", fill = "Power") +
    theme_bw()
}
plot_radial_profile <- function(kymo_df) {
  kymo_df %>% group_by(r) %>% summarize(I_mean = mean(I, na.rm = TRUE), .groups="drop") %>%
    mutate(r_um = r*pixel_size_um) %>%
    ggplot(aes(r_um, I_mean)) + geom_line() +
    labs(title = "Radial Mean Intensity", x = "Radius (µm)", y = "Mean intensity") +
    theme_bw()
}
plot_dominant_mode <- function(dom_df) {
  dom_df %>% mutate(r_um = r*pixel_size_um) %>%
    ggplot(aes(r_um, dom_m, size = ani_idx, color = ani_idx)) +
    geom_point() + labs(title = "Dominant Angular Mode vs Radius", x = "Radius (µm)", y = "Mode (m)", size = "Anisotropy", color = "Anisotropy") +
    theme_bw()
}
plot_rose <- function(img, cx, cy, r_sel, n_theta) {
  circ <- sample_circle_intensity(img, cx, cy, r_sel, n_theta) %>% mutate(theta_deg = theta*180/pi)
  ggplot(circ, aes(x = theta, y = pmax(I,0))) + geom_col(width = 2*pi/n_theta) +
    coord_polar(start = 0) + labs(title = glue("Orientation Rose @ r={r_sel}px"), x = NULL, y = "I") +
    theme_void() + theme(plot.title = element_text(hjust = 0.5))
}

# Per-layer processor: uses nuclear mask from ch1(z,t) and membrane image from chosen channel at same (z,t)
process_one_layer <- function(z, t) {
  nm_mask <- layer_name(nuclear_channel, z, t)
  mk      <- mask_stack[[which(names(mask_stack) == nm_mask)]]
  mkm     <- as.matrix(mk, wide = TRUE)
  
  # centroid from mask
  cxy <- centroid_from_mask(mkm)
  if (any(is.na(cxy))) {
    warning(glue("No nucleus mask at z={z}, t={t}; skipping polar features."))
    return(tibble(z=z, t=t, dom_m_mean = NA, dom_m_max = NA, ani_mean = NA, ani_max = NA, r_max_px = NA))
  }
  cx <- cxy[1]; cy <- cxy[2]
  
  # membrane image at same (z,t)
  nm_mem <- layer_name(membrane_channel, z, t)
  mem_r  <- r[[idx_of(membrane_channel, z, t)]]
  img    <- as.matrix(mem_r, wide = TRUE)
  img[!is.finite(img)] <- 0
  
  nr <- nrow(img); nc <- ncol(img)
  rmax <- if (exists("max_radius_px", inherits = FALSE)) max_radius_px else {
    if (max_radius_mode == "edge") max_radius_to_edge(nr, nc, cx, cy)
    else stop("Set max_radius_px or use max_radius_mode='edge'.")
  }
  r_seq <- seq(min_radius_px, rmax, by = radial_step_px)
  
  # θ–r intensity and angular spectra
  kymo   <- make_kymograph(img, cx, cy, r_seq, n_theta)
  mode_r <- make_mode_radius(img, cx, cy, r_seq, n_theta)
  dom    <- dominant_mode_summary(mode_r, m_exclude0 = TRUE)
  
  feats <- tibble(
    z = z, t = t,
    dom_m_mean = mean(dom$dom_m, na.rm = TRUE),
    dom_m_max  = max(dom$dom_m, na.rm = TRUE),
    ani_mean   = mean(dom$ani_idx, na.rm = TRUE),
    ani_max    = max(dom$ani_idx, na.rm = TRUE),
    r_max_px   = rmax
  )
  
  # save artifacts (limited by t if requested)
  do_save <- if (save_only_t1) t == 1 else TRUE
  if (do_save) {
    base <- file.path(out_dir, glue("z{z}_t{t}"))
    if (save_tables_csv) {
      write_csv(kymo,   glue("{base}_kymograph.csv"))
      write_csv(mode_r, glue("{base}_mode_radius.csv"))
      write_csv(dom,    glue("{base}_dominant.csv"))
    }
    if (save_kymograph_png) {
      g1 <- plot_kymograph(kymo)
      ggsave(glue("{base}_kymograph.png"), g1, width = 6, height = 5, dpi = 300)
    }
    if (save_mode_heat_png) {
      g2 <- plot_mode_heatmap(mode_r)
      ggsave(glue("{base}_mode_heatmap.png"), g2, width = 6, height = 5, dpi = 300)
    }
    if (save_radial_png) {
      g3 <- plot_radial_profile(kymo)
      ggsave(glue("{base}_radial_profile.png"), g3, width = 5, height = 4, dpi = 300)
    }
    if (save_dominant_png) {
      g4 <- plot_dominant_mode(dom)
      ggsave(glue("{base}_dominant_mode.png"), g4, width = 5, height = 4, dpi = 300)
    }
    if (save_rose_png && !is.na(rose_radius_px)) {
      g5 <- plot_rose(img, cx, cy, rose_radius_px, n_theta)
      ggsave(glue("{base}_rose_r{rose_radius_px}.png"), g5, width = 5, height = 5, dpi = 300)
    }
  }
  
  feats
}

# ===========================================================
# 6) RUN: BUILD FEATURES TABLE
# ===========================================================
# features per (z,t) based on the membrane channel and nuclear mask
features_df <- map_dfr(seq_len(n_z), function(z) {
  map_dfr(seq_len(n_time), ~ process_one_layer(z, .x))
}) %>%
  mutate(channel = channel_names[membrane_channel],
         pixel_size_um = pixel_size_um) %>%
  relocate(channel, z, t)

# Save features
write_csv(features_df, file.path(out_dir, "polar_features.csv"))

# Quick QC of anisotropy vs Z at t==1
features_df %>%
  filter(t == 1, !is.na(ani_mean)) %>%
  ggplot(aes(z, ani_mean)) + geom_line() + geom_point() +
  labs(title = "Mean anisotropy vs Z (t=1)", x = "Z", y = "Anisotropy") +
  theme_bw()

# ============================================
# NUCLEUS-CENTRIC HEATMAPS (θ–r and Cartesian)
# ============================================

# ---- choose a layer to visualize ----
z_view <- 16
t_view <- 1

# pull the nuclear mask-derived centroid & membrane image the way your script does
nm_mask <- glue("ch{nuclear_channel}_z{z_view}_t{t_view}")
mk      <- mask_stack[[which(names(mask_stack) == nm_mask)]]
mkm     <- as.matrix(mk, wide = TRUE)

# centroid (must match your helper)
centroid_from_mask <- function(mask_mat) {
  m <- mask_mat; m[is.na(m)] <- 0
  w <- sum(m); if (w <= 0) return(c(NA_real_, NA_real_))
  nr <- nrow(m); nc <- ncol(m)
  xs <- matrix(rep(1:nc, each = nr), nrow = nr)
  ys <- matrix(rep(1:nr, nc), nrow = nr)
  c(sum(xs*m)/w, sum(ys*m)/w)
}
cxy <- centroid_from_mask(mkm); cx <- cxy[1]; cy <- cxy[2]

# membrane image at same (z,t)
mem_r  <- r[[which(names(r) == glue("ch{membrane_channel}_z{z_view}_t{t_view}"))]]
img    <- as.matrix(mem_r, wide = TRUE)
img[!is.finite(img)] <- 0
nr <- nrow(img); nc <- ncol(img)

# replicate your sampling helpers briefly (if not already in scope)
bilinear_sample <- function(img, x, y) {
  nr <- nrow(img); nc <- ncol(img)
  x <- pmin(pmax(x, 1), nc - 1); y <- pmin(pmax(y, 1), nr - 1)
  x0 <- floor(x); x1 <- x0 + 1; y0 <- floor(y); y1 <- y0 + 1
  dx <- x - x0; dy <- y - y0
  i00 <- img[cbind(y0, x0)]; i10 <- img[cbind(y0, x1)]
  i01 <- img[cbind(y1, x0)]; i11 <- img[cbind(y1, x1)]
  i0 <- i00*(1-dx) + i10*dx; i1 <- i01*(1-dx) + i11*dy
  i0*(1-dy) + i1*dy
}
sample_circle_intensity <- function(img, cx, cy, r, n_theta) {
  thetas <- seq(0, 2*pi, length.out = n_theta + 1)[- (n_theta + 1)]
  xs <- cx + r*cos(thetas); ys <- cy + r*sin(thetas)
  tibble(theta = thetas, I = bilinear_sample(img, xs, ys))
}
angular_fft <- function(I_theta) {
  n <- nrow(I_theta); F <- fft(I_theta$I)
  tibble(m = 0:(n-1), F = F, power = (Mod(F)^2)/n) %>% slice(1:(floor(n/2)+1))
}
max_radius_to_edge <- function(nr, nc, cx, cy) floor(min(cx-1, nc-cx, cy-1, nr-cy))

# ---- set polar sampling to match your analysis ----
n_theta_use <- n_theta         # reuse your global setting
rmax <- max_radius_to_edge(nr, nc, cx, cy)
r_seq <- seq(min_radius_px, rmax, by = radial_step_px)

# ---- recompute kymo/mode_r here (or reuse your saved tables) ----
kymo <- purrr::map_dfr(r_seq, ~ mutate(sample_circle_intensity(img, cx, cy, .x, n_theta_use), r = .x))
mode_r <- purrr::map_dfr(r_seq, function(r) {
  spec <- angular_fft(filter(kymo, r == !!r) %>% select(theta, I))
  mutate(spec, r = r)
})
dom <- mode_r %>%
  filter(m > 0) %>%
  group_by(r) %>%
  summarize(
    dom_m   = m[which.max(power)],
    dom_pow = max(power),
    sum_pow = sum(power),
    ani_idx = ifelse(sum_pow > 0, dom_pow/sum_pow, NA_real_), .groups = "drop"
  )

# =========================
# Heatmap A: θ–r kymograph
# =========================
ggplot(kymo %>% mutate(theta_deg = theta*180/pi, r_um = r*pixel_size_um),
       aes(theta_deg, r_um, fill = I)) +
  geom_raster() +
  scale_y_reverse() +
  labs(title = glue("θ–r Kymograph (z={z_view}, t={t_view})"),
       x = expression(theta~"(degrees)"), y = "Radius (µm)", fill = "Intensity") +
  theme_bw()

# ======================================================================
# Heatmap B: θ–r of a SELECTED ANGULAR MODE m_sel (Fourier around circle)
# ======================================================================
m_sel <- 2  # e.g., m=2 for “two-lobed” symmetry; change as needed

# reconstruct mode-m component at each radius: inverse FFT keeping ±m_sel only
reconstruct_mode_component <- function(I_theta_row, m_sel) {
  # I_theta_row: tibble(theta,I) equally spaced
  n <- nrow(I_theta_row)
  F  <- fft(I_theta_row$I)
  # zero all except DC and ±m_sel
  F_filt <- rep(0+0i, n)
  idx_pos <- m_sel + 1L                 # FFT index for +m
  idx_neg <- n - m_sel + 1L             # FFT index for -m
  F_filt[1] <- 0+0i                     # drop DC for pure mode visualization
  F_filt[idx_pos] <- F[idx_pos]
  F_filt[idx_neg] <- F[idx_neg]
  Re(fft(F_filt, inverse = TRUE) / n)   # real part of inverse
}

kymo_mode <- kymo %>%
  group_by(r) %>%
  group_modify(~ {
    comp <- reconstruct_mode_component(.x %>% select(theta, I), m_sel)
    mutate(.x, I_mode = comp)
  }) %>% ungroup()

ggplot(kymo_mode %>% mutate(theta_deg = theta*180/pi, r_um = r*pixel_size_um),
       aes(theta_deg, r_um, fill = I_mode)) +
  geom_raster() +
  scale_y_reverse() +
  labs(title = glue("θ–r Heatmap of Mode m={m_sel} (z={z_view}, t={t_view})"),
       x = expression(theta~"(degrees)"), y = "Radius (µm)", fill = "Mode amplitude") +
  theme_bw()


# Normalize each radius to mean 0, sd 1 for clearer angular contrast
kymo_mode_norm <- kymo_mode %>%
  group_by(r) %>%
  mutate(I_mode_z = (I_mode - mean(I_mode)) / sd(I_mode)) %>%
  ungroup()

ggplot(kymo_mode_norm %>% mutate(theta_deg = theta*180/pi, r_um = r*pixel_size_um),
       aes(theta_deg, r_um, fill = I_mode_z)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_viridis_c(name = "z-score amplitude", option = "plasma") +
  labs(title = "Normalized θ–r Heatmap (mode m=2)",
       x = expression(theta~"(degrees)"), y = "Radius (µm)") +
  theme_bw()

#### this might be Junk ####
nr <- nrow(img)
nc <- ncol(img)

theta_grid <- sort(unique(kymo_mode$theta))
r_grid     <- sort(unique(kymo_mode$r))

# build a matrix of mode amplitudes over (r, θ)
kymo_mat <- kymo_mode %>%
  arrange(r, theta) %>%
  pull(I_mode) %>%
  matrix(nrow = length(r_grid), byrow = TRUE)

# map each pixel (x,y) → nearest (r, θ) bin
Y <- matrix(rep(1:nr, times = nc), nrow = nr)
X <- matrix(rep(1:nc, each = nr), nrow = nr)
R <- sqrt((X - cx)^2 + (Y - cy)^2)
Theta <- (atan2((Y - cy), (X - cx)) + 2*pi) %% (2*pi)

nearest_idx <- function(v, grid) {
  findInterval(v, vec = (head(grid, -1) + tail(grid, -1))/2, all.inside = TRUE) + 1L
}
r_idx     <- nearest_idx(R, r_grid)
theta_idx <- nearest_idx(Theta, theta_grid)

mode_map <- matrix(NA_real_, nrow = nr, ncol = nc)
ok <- R >= min(r_grid) & R <= max(r_grid)
mode_map[ok] <- kymo_mat[cbind(r_idx[ok], theta_idx[ok])]

df_mode <- tibble(
  x = rep(1:nc, each = nr),
  y = rep(1:nr, times = nc),
  val = as.vector(mode_map)    # from the Cartesian projection block
)
ggplot(df_mode, aes(x, y, fill = val)) +
  geom_raster() + coord_equal() + scale_y_reverse() +
  scale_fill_viridis_c(name = "Mode m=2 amplitude") +
  geom_point(aes(x = cx, y = cy), color = "red", size = 2) +
  labs(title = glue("Mode-2 Fourier amplitude (z={z_view}, t={t_view})"),
       x = "X (px)", y = "Y (px)") +
  theme_void()
# ==========================================================
# Heatmap C: Cartesian “anisotropy ring map” + centroid dot
#   - paint the anisotropy index (dominant power fraction) 
#     back into the image as rings centered at the nucleus
# ==========================================================
# Build a 2D map with each pixel assigned ani_idx of its radius bin
make_cartesian_from_radial <- function(nr, nc, cx, cy, r_seq, values_per_r) {
  # values_per_r: tibble r, val  (r must match r_s_
  
# ---------- helpers (safe to re-run) ----------
  bilinear_sample <- function(img, x, y) {
    nr <- nrow(img); nc <- ncol(img)
    x <- pmin(pmax(x, 1), nc - 1); y <- pmin(pmax(y, 1), nr - 1)
    x0 <- floor(x); x1 <- x0 + 1; y0 <- floor(y); y1 <- y0 + 1
    dx <- x - x0; dy <- y - y0
    i00 <- img[cbind(y0, x0)]; i10 <- img[cbind(y0, x1)]
    i01 <- img[cbind(y1, x0)]; i11 <- img[cbind(y1, x1)]
    i0 <- i00*(1-dx) + i10*dx; i1 <- i01*(1-dx) + i11*dy
    i0*(1-dy) + i1*dy
  }
  sample_circle_intensity <- function(img, cx, cy, r, n_theta = 360) {
    thetas <- seq(0, 2*pi, length.out = n_theta + 1)[- (n_theta + 1)]
    xs <- cx + r*cos(thetas); ys <- cy + r*sin(thetas)
    tibble(theta = thetas, I = bilinear_sample(img, xs, ys))
  }
  angular_fft <- function(I_theta) {
    n <- nrow(I_theta); F <- fft(I_theta$I)
    tibble(m = 0:(n-1), F = F, power = (Mod(F)^2)/n) %>% slice(1:(floor(n/2)+1))
  }
  reconstruct_mode_component <- function(I_theta_row, m_sel) {
    n <- nrow(I_theta_row); F <- fft(I_theta_row$I)
    F_filt <- rep(0+0i, n)
    idx_pos <- m_sel + 1L; idx_neg <- n - m_sel + 1L
    F_filt[idx_pos] <- F[idx_pos]; F_filt[idx_neg] <- F[idx_neg]
    Re(fft(F_filt, inverse = TRUE) / n)
  }
  max_radius_to_edge <- function(nr, nc, cx, cy) floor(min(cx-1, nc-cx, cy-1, nr-cy))
  nearest_idx <- function(v, grid) findInterval(v, vec = (head(grid, -1) + tail(grid, -1))/2, all.inside = TRUE) + 1L
  
  # ---------- rebuild mode_map if needed ----------
  m_sel <- 2
  if (!exists("mode_map", inherits = FALSE)) {
    nr <- nrow(img); nc <- ncol(img)
    rmax <- max_radius_to_edge(nr, nc, cx, cy)
    r_seq <- seq(2, rmax, by = 1)        # use your radial_step/min_radius if different
    n_theta_use <- 360                   # use your n_theta if different
    
    # kymograph of raw intensity
    kymo <- purrr::map_dfr(r_seq, ~ mutate(sample_circle_intensity(img, cx, cy, .x, n_theta_use), r = .x))
    
    # mode-m component at each radius
    kymo_mode <- kymo %>%
      group_by(r) %>%
      group_modify(~ {
        comp <- reconstruct_mode_component(.x %>% select(theta, I), m_sel)
        mutate(.x, I_mode = comp)
      }) %>% ungroup()
    
    # project (r, theta) -> (x, y)
    theta_grid <- sort(unique(kymo_mode$theta))
    r_grid     <- sort(unique(kymo_mode$r))
    kymo_mat <- kymo_mode %>% arrange(r, theta) %>% pull(I_mode) %>%
      matrix(nrow = length(r_grid), byrow = TRUE)
    
    Y <- matrix(rep(1:nr, times = nc), nrow = nr)
    X <- matrix(rep(1:nc, each = nr), nrow = nr)
    R <- sqrt((X - cx)^2 + (Y - cy)^2)
    Theta <- (atan2((Y - cy), (X - cx)) + 2*pi) %% (2*pi)
    
    r_idx     <- nearest_idx(R, r_grid)
    theta_idx <- nearest_idx(Theta, theta_grid)
    
    mode_map <- matrix(NA_real_, nrow = nr, ncol = nc)
    ok <- R >= min(r_grid) & R <= max(r_grid)
    mode_map[ok] <- kymo_mat[cbind(r_idx[ok], theta_idx[ok])]
  }
  
  # ---------- build plotting data ----------
  nr <- nrow(img); nc <- ncol(img)
  
  df_orig <- tibble(
    x = rep(1:nc, each = nr),
    y = rep(1:nr, times = nc),
    I = as.vector(img)
  )
  # clip & rescale raw intensity for a nice background
  qI <- quantile(df_orig$I, probs = c(0.02, 0.98), na.rm = TRUE)
  df_orig <- df_orig %>%
    mutate(Ic  = pmin(pmax(I, qI[1]), qI[2]),
           I01 = scales::rescale(Ic, to = c(0, 1)))
  
  df_mode <- tibble(
    x = rep(1:nc, each = nr),
    y = rep(1:nr, times = nc),
    val = as.vector(mode_map)
  )
  # optional clipping of mode amplitudes for nicer contours
  rng <- quantile(df_mode$val, c(0.02, 0.98), na.rm = TRUE)
  
  # ---------- overlay #1: contours on grayscale background (no extra packages) ----------
  # keep only finite (x,y,val) and inside the valid radius
  df_mode_plot <- df_mode %>%
    filter(is.finite(val))                    # drop NA/Inf points
  
  # optional: clip to robust range for prettier contours
  rng <- quantile(df_mode_plot$val, c(0.02, 0.98), na.rm = TRUE)
  df_mode_plot <- df_mode_plot %>%
    mutate(val_clipped = pmin(pmax(val, rng[1]), rng[2]))
  
  # background (unchanged)
  p_contour <-ggplot() +
    geom_raster(data = df_orig, aes(x, y, color = I01)) +
    scale_color_gradient(low = "black", high = "white", guide = "none") +
    geom_raster(data = df_mode_plot, aes(x, y, fill = val_clipped), alpha = 0.55) +
    scale_fill_viridis_c(name = "m=2 amplitude") +
    coord_equal() + scale_y_reverse() +
    geom_point(aes(x = cx, y = cy), color = "red", size = 2) +
    theme_void() + theme(legend.position = "right")
  print(p_contour)
  # ggsave(glue("overlay_mode2_contours_z{z_view}_t{t_view}.png"), p_contour, width = 5.5, height = 5, dpi = 300)
  
  mode_profile <- tibble(
    r_um = r_grid * pixel_size_um,
    amp  = apply(V, 1, function(x) mean(abs(x)))
  )
  ggplot(mode_profile, aes(r_um, amp)) +
    geom_line() + theme_bw() +
    labs(x = "Radius (µm)", y = "|m=2| amplitude",
         title = "Radial strength of 2-fold symmetry")
  
  kymo_phase <- kymo_mode %>%
    group_by(r) %>%
    summarise(phase = atan2(Im(fft(I)), Re(fft(I)))[m_sel+1])
  # ---------- overlay #2 (optional): semi-transparent heatmap on top ----------
  # If you have ggnewscale installed, you can do a dual-scale fill overlay.
  # Otherwise, this uses color for the background and fill for the heatmap.
  
  p_heat <- ggplot() +
    # background in grayscale (use color aesthetic to avoid fill conflicts)
    geom_raster(data = df_orig, aes(x, y, color = I01)) +
    scale_color_gradient(low = "black", high = "white", guide = "none") +
    # heatmap of mode amplitude (signed)
    geom_raster(data = df_mode, aes(x, y, fill = val), alpha = 0.55, na.rm = TRUE) +
    scale_fill_viridis_c(name = "m=2 amplitude",
                         limits = c(rng[1], rng[2]), oob = scales::squish) +
    coord_equal() + scale_y_reverse() +
    geom_point(aes(x = cx, y = cy), color = "red", size = 2) +
    labs(title = glue("Membrane + Mode-2 heatmap (z={z_view}, t={t_view})"),
         x = "X (px)", y = "Y (px)") +
    theme_void() + theme(legend.position = "right")
  
  print(p_heat)
  # ggsave(glue("overlay_mode2_heat_z{z_view}_t{t_view}.png"), p_heat, width = 5.5, height = 5, dpi = 300)