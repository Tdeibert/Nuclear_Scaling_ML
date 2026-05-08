# ===========================================================
# 0) PACKAGES
# ===========================================================
library(terra)
library(tidyverse)   # dplyr, tidyr, purrr, tibble, stringr, ggplot2, readr
library(glue)
library(scales)
library(ggnewscale)
library(zoo)

# ===========================================================
# 1) USER SETTINGS (EDIT THESE)
# ===========================================================
# ----- Data & geometry -----
stack_path      <- "C:/Users/tdeibert/.Working_Docs_Folder/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Region 1/Droplet 1/Region 1 Droplet 1 .tif"
n_channels      <- 3
n_z             <- 20
n_time          <- 10
channel_names   <- c("membrane","NLS","Alexaflour647")
pixel_size_um   <- 0.108  # μm/px

# ----- Channel roles -----
nuclear_channel  <- 2   # NLS channel index (1..n_channels)
membrane_channel <- 1   # membrane channel index

# ----- Output -----
out_dir          <- "nucleus_mask_and_polar_outputs"
dir.create(out_dir, showWarnings = FALSE)

# ----- Masking params -----
gauss_sigma_px   <- 1.5
gauss_radius_px  <- ceiling(gauss_sigma_px*3)
otsu_multiplier  <- 1.00              # <1 expands, >1 tightens
min_obj_area_um2 <- 25
min_obj_area_px  <- ceiling(min_obj_area_um2 / (pixel_size_um^2))
keep_largest     <- TRUE
fill_holes       <- TRUE
save_mask_qc_png <- TRUE              # saves for t==1 only (see below)

# ----- Polar analysis params -----
n_theta          <- 360               # angular samples (deg resolution)
radial_step_px   <- 1
min_radius_px    <- 2
max_radius_mode  <- "edge"            # or set max_radius_px

# ----- Focus selection params -----
# Composite focus = 0.6·Tenengrad(inner rim) + 0.3·DoG(inner rim) + 0.1·(Ten_inner − Ten_outer)
ring_w_in        <- 3                 # inner rim width  (px)
ring_off_in      <- 0                 # inner rim offset (px)
ring_w_out       <- 3                 # outer rim width  (px)
ring_off_out     <- 0                 # outer rim offset (px)
ten_wt           <- 0.6
dog_wt           <- 0.3
ed_wt            <- 0.1

# Temporal smoothing of Z choices (Viterbi-style path with jump penalty)
jump_penalty     <- 0.35              # ↑ smoother path (try 0.25–0.6)
max_step         <- NULL              # e.g., 2 to forbid big jumps (optional)

# Optional: Quadratic refinement around local peak before snapping to int Z
use_quadratic_refine <- TRUE

# Toggle artifact saving
save_tables_csv  <- TRUE
save_kymograph_png <- TRUE
save_mode_heat_png  <- TRUE
save_radial_png     <- TRUE
save_dominant_png   <- TRUE
save_rose_png       <- TRUE
rose_radius_px      <- 30

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

# Helpers to get layer names / indices
layer_name <- function(ch, z, t) glue("ch{ch}_z{z}_t{t}")
idx_of     <- function(ch, z, t) which(names(r) == layer_name(ch,z,t))

# ===========================================================
# 3) LOW-LEVEL HELPERS (image ops, morphology, FFT, etc.)
# ===========================================================
# -- kernels, filters --
gaussian_kernel <- function(sigma, radius) {
  xs <- -radius:radius
  g1 <- exp(-(xs^2) / (2*sigma^2))
  ker <- outer(g1, g1, "*")
  ker / sum(ker)
}
gauss <- function(sigma, radius = ceiling(3*sigma)) {
  xs <- -radius:radius
  g1 <- exp(-(xs^2)/(2*sigma^2))
  ker <- outer(g1, g1, "*"); ker/sum(ker)
}
conv2 <- function(r, ker) focal(r, w = ker, fun = "sum", na.policy = "omit", na.rm = TRUE)

sobel_x <- matrix(c(-1,0,1,-2,0,2,-1,0,1), 3, 3, byrow=TRUE)
sobel_y <- matrix(c(-1,-2,-1,0,0,0,1,2,1), 3, 3, byrow=TRUE)

# -- thresholding --
otsu_threshold <- function(v) {
  v <- v[is.finite(v)]
  v <- v[v >= quantile(v, 0.001) & v <= quantile(v, 0.999)]
  if (length(v) < 256) return(mean(v, na.rm=TRUE))
  h <- hist(v, breaks = 256, plot = FALSE)
  p <- h$counts / sum(h$counts)
  omega <- cumsum(p)
  mu <- cumsum(p * h$mids)
  mu_t <- mu[length(mu)]
  sigma_b2 <- (mu_t * omega - mu)^2 / (omega * (1 - omega) + 1e-12)
  h$mids[which.max(replace(sigma_b2, !is.finite(sigma_b2), -Inf))]
}

# -- morphology (binary) --
morph_kernel <- function(k) matrix(1, nrow = (2*k+1), ncol = (2*k+1))
dilate_px <- function(bin_r, k) {
  if (k <= 0) return(bin_r)
  ker <- morph_kernel(k)
  s   <- focal(bin_r, w = ker, fun = "sum", na.policy = "omit", na.rm = TRUE)
  clamp(s > 0, lower=0, upper=1, values=TRUE)
}
erode_px <- function(bin_r, k) {
  if (k <= 0) return(bin_r)
  ker <- morph_kernel(k); full <- (2*k+1)^2
  s   <- focal(bin_r, w = ker, fun = "sum", na.policy = "omit", na.rm = TRUE)
  clamp(s == full, lower=0, upper=1, values=TRUE)
}
fill_binary_holes <- function(bin_r) {
  p <- as.polygons(bin_r, dissolve = TRUE, values = TRUE, trunc = TRUE)
  if (nrow(p) == 0) return(bin_r)
  r_out <- rast(bin_r)
  rasterize(p, r_out, field = 1, background = 0, touches = TRUE)
}

# -- rings from mask --
inner_ring <- function(mask_r, offset=0, width=3) {
  inner1 <- erode_px(mask_r, offset)
  inner2 <- erode_px(mask_r, offset + width)
  clamp(inner1 - inner2, lower=0, upper=1, values=TRUE)
}
outer_ring <- function(mask_r, offset=0, width=3) {
  out1 <- dilate_px(mask_r, offset + width)
  out2 <- dilate_px(mask_r, offset)
  clamp(out1 - out2, lower=0, upper=1, values=TRUE)
}

# -- focus metrics --
tenengrad_energy <- function(img_r) {
  gx <- conv2(img_r, sobel_x); gy <- conv2(img_r, sobel_y)
  (gx*gx + gy*gy)
}
dog_filter <- function(img_r, s1 = 1.0, s2 = 2.5) {
  g1 <- conv2(img_r, gauss(s1)); g2 <- conv2(img_r, gauss(s2))
  g1 - g2
}
sum_in_roi <- function(val_r, roi_r) {
  v <- values(val_r, mat = FALSE); m <- values(roi_r, mat = FALSE)
  n <- min(length(v), length(m))
  sum(v[seq_len(n)] * (m[seq_len(n)] > 0), na.rm = TRUE)
}

# -- centroid from binary mask (matrix) --
centroid_from_mask <- function(mask_mat) {
  m <- mask_mat; m[is.na(m)] <- 0
  w <- sum(m); if (w <= 0) return(c(NA_real_, NA_real_))
  nr <- nrow(m); nc <- ncol(m)
  xs <- matrix(rep(1:nc, each = nr), nrow = nr)
  ys <- matrix(rep(1:nr, nc), nrow = nr)
  c(sum(xs*m)/w, sum(ys*m)/w)
}

# -- sampling on circles + angular FFT --
bilinear_sample <- function(img, x, y) {
  nr <- nrow(img); nc <- ncol(img)
  x <- pmin(pmax(x, 1), nc - 1); y <- pmin(pmax(y, 1), nr - 1)
  x0 <- floor(x); x1 <- x0 + 1; y0 <- floor(y); y1 <- y0 + 1
  dx <- x - x0; dy <- y - y0
  i00 <- img[cbind(y0, x0)]; i10 <- img[cbind(y0, x1)]
  i01 <- img[cbind(y1, x0)]; i11 <- img[cbind(y1, x1)]
  i0 <- i00*(1-dx) + i10*dx
  i1 <- i01*(1-dx) + i11*dx
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
max_radius_to_edge <- function(nr, nc, cx, cy) floor(min(cx-1, nc-cx, cy-1, nr-cy))

# ===========================================================
# 4) NUCLEAR MASKS (from NLS channel) + QC overlays (t==1)
# ===========================================================
gker <- gaussian_kernel(gauss_sigma_px, gauss_radius_px)

derive_nuclear_mask <- function(nls_layer_r) {
  sm  <- focal(nls_layer_r, w = gker, fun = "sum", na.policy = "omit", na.rm = TRUE)
  v   <- values(sm, mat=FALSE)
  m   <- median(v, na.rm=TRUE); s <- mad(v, constant = 1, na.rm=TRUE) + 1e-9
  zn  <- (sm - m) / s
  
  zmat <- as.matrix(zn, wide = TRUE)
  thr  <- otsu_threshold(zmat)*otsu_multiplier
  bin  <- zmat > thr
  
  bin_r <- rast(bin); ext(bin_r) <- ext(nls_layer_r); crs(bin_r) <- crs(nls_layer_r)
  cl    <- patches(bin_r, directions = 8, zeroAsNA = TRUE)
  
  if (!is.null(min_obj_area_px) && is.finite(min_obj_area_px) && min_obj_area_px > 1) {
    ff   <- freq(cl) %>% as_tibble() %>% rename(label = value, npx = count)
    keep <- ff %>% filter(npx >= min_obj_area_px) %>% pull(label)
    cl   <- classify(cl, rcl = cbind(setdiff(unique(values(cl)), keep), NA))
  }
  if (keep_largest) {
    ff <- freq(cl) %>% as_tibble() %>% rename(label = value, npx = count)
    if (nrow(ff) > 0) {
      lab <- ff$label[which.max(ff$npx)]
      cl  <- classify(cl, rcl = cbind(setdiff(unique(values(cl)), lab), NA))
    }
  }
  mask <- clamp(!is.na(cl), lower=0, upper=1, values=TRUE)
  if (fill_holes) mask <- fill_binary_holes(mask)
  mask
}

message("Deriving nuclear masks from channel: ", nuclear_channel)
mask_stack <- rast()
for (z in seq_len(n_z)) {
  for (t in seq_len(n_time)) {
    idx_nls <- idx_of(nuclear_channel, z, t)
    nls_r   <- r[[idx_nls]]
    mk      <- derive_nuclear_mask(nls_r)
    names(mk) <- names(nls_r)
    mask_stack <- c(mask_stack, mk)
  }
}

if (save_mask_qc_png) {
  qc_dir <- file.path(out_dir, "mask_qc")
  dir.create(qc_dir, showWarnings = FALSE)
  for (z in seq_len(n_z)) {
    t <- 1
    nm    <- layer_name(nuclear_channel, z, t)
    nls_r <- r[[idx_of(nuclear_channel, z, t)]]
    mk    <- mask_stack[[which(names(mask_stack) == nm)]]
    
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
# 5) COMPOSITE FOCUS SCORE → TEMPORALLY SMOOTH Z PATH
# ===========================================================
# Compute focus cues per (z,t) on a thin rim near the nuclear boundary
focus_scores <- purrr::map_dfr(seq_len(n_time), function(t) {
  purrr::map_dfr(seq_len(n_z), function(z) {
    tryCatch({
      nm_mask <- layer_name(nuclear_channel, z, t)
      mk      <- mask_stack[[which(names(mask_stack) == nm_mask)]]
      if (is.null(mk)) return(tibble(t=t, z=z, ten_in=NA, dog_in=NA, ed_contrast=NA, tag="no-mask"))
      
      mkm <- values(mk, mat = FALSE)
      if (all(!is.finite(mkm)) || sum(mkm, na.rm = TRUE) == 0) {
        return(tibble(t=t, z=z, ten_in=NA, dog_in=NA, ed_contrast=NA, tag="empty-mask"))
      }
      
      rim_in_r  <- inner_ring(mk,  offset = ring_off_in,  width = ring_w_in)
      rim_out_r <- outer_ring(mk,  offset = ring_off_out, width = ring_w_out)
      
      mem_r <- r[[idx_of(membrane_channel, z, t)]]
      mem_r[!is.finite(mem_r)] <- 0
      
      ten <- tenengrad_energy(mem_r)
      dog <- abs(dog_filter(mem_r, s1 = 1.0, s2 = 2.5))
      
      ten_in <- sum_in_roi(ten, rim_in_r)
      dog_in <- sum_in_roi(dog, rim_in_r)
      ten_out <- sum_in_roi(ten, rim_out_r)
      
      tibble(t=t, z=z,
             ten_in=ten_in,
             dog_in=dog_in,
             ed_contrast = ten_in - ten_out,
             tag="ok")
    }, error = function(e) tibble(t=t, z=z, ten_in=NA, dog_in=NA, ed_contrast=NA, tag="err"))
  })
}) %>%
  group_by(t) %>%
  mutate(
    ten_z  = (ten_in  - median(ten_in,  na.rm=TRUE)) / (mad(ten_in,  constant=1, na.rm=TRUE) + 1e-9),
    dog_z  = (dog_in  - median(dog_in,  na.rm=TRUE)) / (mad(dog_in,  constant=1, na.rm=TRUE) + 1e-9),
    ed_z   = (ed_contrast - median(ed_contrast, na.rm=TRUE)) / (mad(ed_contrast, constant=1, na.rm=TRUE) + 1e-9),
    focus  = ten_wt*ten_z + dog_wt*dog_z + ed_wt*ed_z
  ) %>% ungroup()

# Fallback: if a time has all NAs, fall back to whole-frame Tenengrad
fallback_whole <- purrr::map_dfr(seq_len(n_time), function(t) {
  sl <- focus_scores %>% filter(t == !!t)
  if (all(!is.finite(sl$focus))) {
    purrr::map_dfr(seq_len(n_z), function(z) {
      mem_r <- r[[idx_of(membrane_channel, z, t)]]
      mem_r[!is.finite(mem_r)] <- 0
      ten   <- tenengrad_energy(mem_r)
      tibble(t=t, z=z, focus_fb = global(ten, "sum", na.rm=TRUE)[[1]])
    }) %>%
      mutate(focus_fb_z = (focus_fb - median(focus_fb, na.rm=TRUE)) /
               (mad(focus_fb, constant=1, na.rm=TRUE) + 1e-9)) %>%
      select(t, z, focus_fb_z)
  } else tibble(t=t, z=integer(), focus_fb_z=numeric())
}) %>% bind_rows()

focus_scores2 <- focus_scores %>%
  left_join(fallback_whole, by = c("t","z")) %>%
  mutate(focus_final = dplyr::coalesce(focus, focus_fb_z))

# ----- Build F[z,t] and normalize per time -----
nz <- n_z; nt <- n_time
F <- matrix(-Inf, nrow = nz, ncol = nt)
fs <- focus_scores2 %>% filter(is.finite(focus_final))
F[cbind(fs$z, fs$t)] <- fs$focus_final
for (tt in seq_len(nt)) {
  col <- F[, tt]
  if (any(is.finite(col))) {
    m <- median(col[is.finite(col)], na.rm=TRUE)
    s <- mad(col[is.finite(col)], constant=1, na.rm=TRUE) + 1e-9
    F[, tt] <- (col - m)/s
  }
}

# ----- Viterbi-like smoothing with jump penalty -----
dp  <- matrix(-Inf, nrow = nz, ncol = nt)
bp  <- matrix(NA_integer_, nrow = nz, ncol = nt)
dp[,1] <- F[,1]
for (tt in 2:nt) {
  for (z in 1:nz) {
    prev_zs <- 1:nz
    if (!is.null(max_step)) prev_zs <- prev_zs[abs(prev_zs - z) <= max_step]
    scores <- dp[prev_zs, tt-1] - jump_penalty * abs(z - prev_zs)
    best_i <- which.max(scores)
    dp[z, tt] <- F[z, tt] + scores[best_i]
    bp[z, tt] <- prev_zs[best_i]
  }
}
best_z_path <- integer(nt)
best_z_path[nt] <- which.max(dp[, nt])
for (tt in nt:2) best_z_path[tt-1] <- bp[best_z_path[tt], tt]

# ----- Optional quadratic peak refinement (3-point parabola) -----
refine_quadratic <- function(Fcol, z0) {
  if (!use_quadratic_refine) return(z0)
  if (z0 <= 1 || z0 >= length(Fcol) || any(!is.finite(Fcol[(z0-1):(z0+1)]))) return(z0)
  y1 <- Fcol[z0-1]; y2 <- Fcol[z0]; y3 <- Fcol[z0+1]
  denom <- (y1 - 2*y2 + y3)
  if (abs(denom) < 1e-12) return(z0)
  delta <- 0.5 * (y1 - y3) / denom
  z_star <- z0 + delta
  round(pmin(pmax(z_star, 1), length(Fcol)))  # snap to nearest slice
}
best_z_refined <- map_int(seq_len(nt), function(tt) refine_quadratic(F[, tt], best_z_path[tt]))

mode_int <- function(x) {
  tx <- table(x)
  as.integer(names(tx)[which.max(tx)])
}

best_z_per_t <- tibble(t = seq_len(nt), best_z = best_z_refined) %>%
  mutate(
    best_z_smooth = zoo::rollapply(
      best_z,
      width   = 3,
      FUN     = mode_int,
      align   = "center",
      partial = TRUE
    ),
    # for the edges, rollapply with partial=TRUE yields values; if any NA slip in, fall back:
    best_z_smooth = dplyr::coalesce(best_z_smooth, best_z)
  )
# ===========================================================
# 6) POLAR FOURIER FEATURES (membrane channel)
# ===========================================================
process_one_layer <- function(z, t) {
  nm_mask <- layer_name(nuclear_channel, z, t)
  mk      <- mask_stack[[which(names(mask_stack) == nm_mask)]]
  mkm     <- as.matrix(mk, wide = TRUE)
  
  cxy <- centroid_from_mask(mkm)
  if (any(is.na(cxy))) {
    return(tibble(z=z, t=t, dom_m_mean = NA, dom_m_max = NA,
                  ani_mean = NA, ani_max = NA, r_max_px = NA))
  }
  cx <- cxy[1]; cy <- cxy[2]
  
  mem_r <- r[[idx_of(membrane_channel, z, t)]]
  img   <- as.matrix(mem_r, wide = TRUE)
  img[!is.finite(img)] <- 0
  
  nr <- nrow(img); nc <- ncol(img)
  rmax <- if (exists("max_radius_px", inherits = FALSE)) max_radius_px else {
    if (max_radius_mode == "edge") max_radius_to_edge(nr, nc, cx, cy)
    else stop("Set max_radius_px or use max_radius_mode='edge'.")
  }
  r_seq <- seq(min_radius_px, rmax, by = radial_step_px)
  
  kymo   <- map_dfr(r_seq, ~ mutate(sample_circle_intensity(img, cx, cy, .x, n_theta), r = .x))
  mode_r <- map_dfr(r_seq, ~ mutate(angular_fft(filter(kymo, r == .x) %>% select(theta, I)), r = .x))
  
  dom <- mode_r %>%
    filter(m > 0) %>%
    group_by(r) %>%
    summarize(
      dom_m   = m[which.max(power)],
      dom_pow = max(power),
      sum_pow = sum(power),
      ani_idx = ifelse(sum_pow > 0, dom_pow/sum_pow, NA_real_), .groups = "drop"
    )
  
  feats <- tibble(
    z = z, t = t,
    dom_m_mean = mean(dom$dom_m, na.rm = TRUE),
    dom_m_max  = max(dom$dom_m, na.rm = TRUE),
    ani_mean   = mean(dom$ani_idx, na.rm = TRUE),
    ani_max    = max(dom$ani_idx, na.rm = TRUE),
    r_max_px   = rmax
  )
  
  # Optional artifacts (limit by t==1 to reduce I/O)
  if (t == 1) {
    base <- file.path(out_dir, glue("z{z}_t{t}"))
    if (save_tables_csv) {
      write_csv(kymo,   glue("{base}_kymograph.csv"))
      write_csv(mode_r, glue("{base}_mode_radius.csv"))
      write_csv(dom,    glue("{base}_dominant.csv"))
    }
    if (save_kymograph_png) {
      g1 <- kymo %>%
        mutate(theta_deg = theta*180/pi, r_um = r*pixel_size_um) %>%
        ggplot(aes(theta_deg, r_um, fill = I)) +
        geom_raster() + scale_y_reverse() +
        labs(title = "θ–r Intensity (Kymograph)",
             x = expression(theta~"(deg)"), y = "Radius (µm)", fill = "I") +
        theme_bw()
      ggsave(glue("{base}_kymograph.png"), g1, width = 6, height = 5, dpi = 300)
    }
    if (save_mode_heat_png) {
      g2 <- mode_r %>%
        mutate(r_um = r*pixel_size_um) %>%
        ggplot(aes(m, r_um, fill = power)) +
        geom_raster() +
        labs(title = "Angular Fourier Power vs Radius",
             x = "Mode (m)", y = "Radius (µm)", fill = "Power") +
        theme_bw()
      ggsave(glue("{base}_mode_heatmap.png"), g2, width = 6, height = 5, dpi = 300)
    }
    if (save_radial_png) {
      g3 <- kymo %>% group_by(r) %>% summarize(I_mean = mean(I, na.rm = TRUE), .groups="drop") %>%
        mutate(r_um = r*pixel_size_um) %>%
        ggplot(aes(r_um, I_mean)) + geom_line() +
        labs(title = "Radial Mean Intensity", x = "Radius (µm)", y = "Mean intensity") +
        theme_bw()
      ggsave(glue("{base}_radial_profile.png"), g3, width = 5, height = 4, dpi = 300)
    }
    if (save_dominant_png) {
      g4 <- dom %>% mutate(r_um = r*pixel_size_um) %>%
        ggplot(aes(r_um, dom_m, size = ani_idx, color = ani_idx)) +
        geom_point() +
        labs(title = "Dominant Angular Mode vs Radius",
             x = "Radius (µm)", y = "Mode (m)", size = "Anisotropy", color = "Anisotropy") +
        theme_bw()
      ggsave(glue("{base}_dominant_mode.png"), g4, width = 5, height = 4, dpi = 300)
    }
    if (save_rose_png && !is.na(rose_radius_px)) {
      circ <- sample_circle_intensity(img, cx, cy, rose_radius_px, n_theta) %>%
        mutate(theta_deg = theta*180/pi)
      g5 <- ggplot(circ, aes(x = theta, y = pmax(I,0))) + geom_col(width = 2*pi/n_theta) +
        coord_polar(start = 0) + labs(title = glue("Orientation Rose @ r={rose_radius_px}px"),
                                      x = NULL, y = "I") +
        theme_void() + theme(plot.title = element_text(hjust = 0.5))
      ggsave(glue("{base}_rose_r{rose_radius_px}.png"), g5, width = 5, height = 5, dpi = 300)
    }
  }
  feats
}

features_df <- map_dfr(seq_len(n_z), function(z) {
  map_dfr(seq_len(n_time), ~ process_one_layer(z, .x))
}) %>%
  mutate(channel = channel_names[membrane_channel],
         pixel_size_um = pixel_size_um) %>%
  relocate(channel, z, t)

if (save_tables_csv) write_csv(features_df, file.path(out_dir, "polar_features.csv"))

# ===========================================================
# 7) FINAL PLOT: ANISOTROPY OVER TIME USING CHOSEN Z (LABELED)
# ===========================================================
# Use smoothed refined path
bestZ <- best_z_per_t %>% transmute(t, best_z = best_z_smooth)

ani_best_plane <- features_df %>%
  inner_join(bestZ, by = "t") %>%
  filter(z == best_z) %>%
  arrange(t)

p_final <- ggplot(ani_best_plane, aes(x = t, y = ani_mean)) +
  geom_line(color = "grey60") +
  geom_point(aes(color = factor(best_z)), size = 2.8) +
  geom_text(aes(label = paste0("z=", best_z)), vjust = -0.7, size = 3) +
  scale_color_viridis_d(name = "Z used") +
  labs(
    title    = "Anisotropy over time (best-focus Z per time, smoothed)",
    subtitle = glue("Composite focus: {ten_wt}·Ten + {dog_wt}·DoG + {ed_wt}·(Ten_in−Ten_out),  Viterbi λ={jump_penalty}"),
    x = "Time index (t)", y = "Mean anisotropy"
  ) +
  theme_bw()

print(p_final)
ggsave(file.path(out_dir, "anisotropy_over_time_bestZ_labeled.png"),
       p_final, width = 7, height = 4.5, dpi = 300)

# ===========================================================
# 8) OPTIONAL QUICK DIAGNOSTICS
# ===========================================================
if (save_tables_csv) {
  write_csv(focus_scores2, file.path(out_dir, "focus_scores_composite_by_z_t.csv"))
  write_csv(best_z_per_t, file.path(out_dir, "best_z_path_refined_and_smoothed.csv"))
}

# Quick check: anisotropy vs Z at t=1 (raw, unsliced)
features_df %>%
  filter(t == 1, !is.na(ani_mean)) %>%
  ggplot(aes(z, ani_mean)) + geom_line() + geom_point() +
  labs(title = "Mean anisotropy vs Z (t=1)", x = "Z", y = "Anisotropy") +
  theme_bw()
