# ---- Packages ----
library(terra)
library(tidyverse)
library(glue)
library(h2o)
library(imager)
library(broom)

# ==== 0) Metadata ====
# Adjust to your dataset
n_channels <- 3
n_z        <- 20
n_time     <- 2
channel_names <- c("ch1","ch2","ch3")  # put your real channel names here
pixel_size_um <- 0.108                 # microns per pixel (example; set yours!)

# ==== 1) Load your raster stack ====
# Expecting a SpatRaster with nlyr = n_channels * n_z * n_time
r <- rast("C:/Users/tdeibert/.Working_Docs_Folder/UWYO/Nuclear Scaling Project/Code Repository/nuclear-scaling/Test_stack.tif") 
stopifnot(nlyr(r) == n_channels * n_z * n_time)

# Assign clean layer names: ch{c}_z{z}_t{t}
layer_index <- expand_grid(
  ch = seq_len(n_channels),
  z  = seq_len(n_z),
  t  = seq_len(n_time)
) %>%
  mutate(name = str_glue("ch{ch}_z{z}_t{t}"))

names(r) <- layer_index$name

# (Optional) per-layer ROI mask stack (same geometry as r)
mask_r <- NULL  # set to a SpatRaster if you have masks

# ======================
# Helpers
# ======================
fftshift <- function(m){
  nr <- nrow(m); nc <- ncol(m)
  r <- ceiling(nr/2); c <- ceiling(nc/2)
  m[c((r+1):nr, 1:r), c((c+1):nc, 1:c)]
}

radial_profile <- function(power_mat, n_bins = 128) {
  nr <- nrow(power_mat); nc <- ncol(power_mat)
  cx <- (nc+1)/2; cy <- (nr+1)/2
  xs <- matrix(rep(1:nc, each = nr), nrow = nr)
  ys <- matrix(rep(1:nr, nc), nrow = nr)
  r  <- sqrt((xs - cx)^2 + (ys - cy)^2)
  
  tibble(
    freq_pix = as.vector(r) / max(cx-1, cy-1),   # normalized [0..1]
    pwr      = as.vector(power_mat)
  ) %>%
    mutate(bin = cut(freq_pix, breaks = seq(0, 1, length.out = n_bins+1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarize(freq_pix = mean(freq_pix), pwr = mean(pwr), .groups = "drop")
}

pix_to_um_freq <- function(freq_pix, pixel_size_um) {
  # Nyquist (freq_pix = 1.0) == 1/(2*pixel_size)
  freq_pix * (1/(2*pixel_size_um))
}

make_spectral_features <- function(Psh, ps_1d) {
  total_power <- sum(ps_1d$pwr)
  
  band_power <- function(df, lo, hi) {
    df %>% filter(freq_pix >= lo, freq_pix < hi) %>%
      summarize(bp = sum(pwr), .groups="drop") %>% pull(bp)
  }
  
  # tune band edges to your biology
  b1 <- c(0.00, 0.15)
  b2 <- c(0.15, 0.35)
  b3 <- c(0.35, 1.00)
  
  slope <- ps_1d %>%
    filter(freq_pix > 0, pwr > 0) %>%
    transmute(lf = log(freq_pix), lp = log(pwr)) %>%
    lm(lp ~ lf, data = .) %>%
    tidy() %>% filter(term == "lf") %>% pull(estimate)
  
  # crude anisotropy: row vs col cross near DC
  nr <- nrow(Psh); nc <- ncol(Psh); cxy <- c(ceiling(nr/2), ceiling(nc/2))
  k <- max(2, floor(min(nr, nc) * 0.02))
  row_band <- Psh[(cxy[1]-k):(cxy[1]+k), , drop = FALSE]
  col_band <- Psh[ , (cxy[2]-k):(cxy[2]+k), drop = FALSE]
  ani <- (sum(row_band) + 1e-12) / (sum(col_band) + 1e-12)
  
  tibble(
    bp_low        = band_power(ps_1d, b1[1], b1[2]) / total_power,
    bp_mid        = band_power(ps_1d, b2[1], b2[2]) / total_power,
    bp_high       = band_power(ps_1d, b3[1], b3[2]) / total_power,
    spec_centroid = sum(ps_1d$freq_pix * ps_1d$pwr) / total_power,
    spec_slope    = slope,
    anisotropy    = ani,
    total_power   = total_power
  )
}

# ======================
# Plotting helpers (optional)
# ======================
plot_fourier_2d <- function(Psh, clip_quant = 0.999) {
  df2d <- as.data.frame(Psh) |>
    tibble::as_tibble() |>
    mutate(y = dplyr::row_number()) |>
    tidyr::pivot_longer(-y, names_to = "x", values_to = "pwr") |>
    mutate(x = as.integer(stringr::str_remove(x, "^V")),
           lp = log10(pmax(1e-12, pwr)))
  hi <- quantile(df2d$lp, clip_quant, na.rm = TRUE)
  df2d$lp[df2d$lp > hi] <- hi
  
  ggplot(df2d, aes(x, y, fill = lp)) +
    geom_raster() + coord_equal() +
    scale_fill_viridis_c(name = "log10 Power") +
    labs(title = "2-D Power Spectrum", x = "fx (pixels⁻¹)", y = "fy (pixels⁻¹)") +
    theme_void() + theme(plot.title = element_text(hjust = 0.5))
}

plot_radial_spectrum <- function(ps_1d, pixel_size_um) {
  ps_1d %>%
    filter(freq_pix > 0, pwr > 0) %>%
    mutate(freq_um = pix_to_um_freq(freq_pix, pixel_size_um)) %>%
    ggplot(aes(freq_um, pwr)) +
    geom_line() +
    scale_x_log10(name = "Spatial frequency (µm⁻¹)") +
    scale_y_log10(name = "Power") +
    ggtitle("Radially Averaged Power Spectrum") +
    theme_bw()
}

# ======================
# Core: compute FFT features with Option B returns
# ======================
compute_fft_features_for_layer <- function(mat_layer, return_ps = FALSE, return_Psh = FALSE) {
  mat <- mat_layer
  mat[is.na(mat)] <- 0
  mat <- mat - mean(mat)
  
  F   <- fft(mat)
  P   <- Mod(F)^2
  Psh <- fftshift(P)
  
  ps_1d <- radial_profile(Psh, n_bins = 128) %>%
    mutate(freq_um = pix_to_um_freq(freq_pix, pixel_size_um))
  
  features <- make_spectral_features(Psh, ps_1d)
  
  if (return_ps || return_Psh) {
    out <- list(features = features)
    if (return_ps)  out$ps  <- ps_1d
    if (return_Psh) out$Psh <- Psh
    return(out)
  }
  features
}

# ======================
# Batch over full raster with optional saving of spectra
# ======================
# Controls for saving (set TRUE to enable)
save_ps_csv      <- TRUE
save_radial_png  <- TRUE
save_Psh_png     <- TRUE

# Save only for a subset to avoid huge I/O (e.g., t == 1), or set to TRUE for all
save_only_t1     <- TRUE

out_dir <- "fourier_outputs"
dir.create(out_dir, showWarnings = FALSE)

process_one_layer <- function(i) {
  nm <- names(r)[i]
  ch <- as.integer(str_match(nm, "ch(\\d+)")[,2])
  z  <- as.integer(str_match(nm, "_z(\\d+)")[,2])
  t  <- as.integer(str_match(nm, "_t(\\d+)")[,2])
  
  m <- as.matrix(r[[i]], wide = TRUE)
  if (!is.null(mask_r)) {
    mk <- as.matrix(mask_r[[i]], wide = TRUE)
    m[is.na(mk) | mk == 0] <- 0
  }
  
  # Decide whether to save plots/CSV for this layer
  do_save <- if (save_only_t1) t == 1 else TRUE
  
  res <- compute_fft_features_for_layer(m, return_ps = do_save, return_Psh = do_save)
  feats <- if (do_save) res$features else res
  
  # Save artifacts if requested
  if (do_save && save_ps_csv) {
    write_csv(res$ps, file.path(out_dir, glue("{nm}_ps1d.csv")))
  }
  if (do_save && save_radial_png) {
    g <- plot_radial_spectrum(res$ps, pixel_size_um)
    ggsave(filename = file.path(out_dir, glue("{nm}_radial.png")), plot = g, width = 5, height = 4, dpi = 300)
  }
  if (do_save && save_Psh_png) {
    g2 <- plot_fourier_2d(res$Psh)
    ggsave(filename = file.path(out_dir, glue("{nm}_Psh.png")), plot = g2, width = 5, height = 5, dpi = 300)
  }
  
  tibble(
    channel = channel_names[ch],
    ch = ch, z = z, t = t
  ) %>%
    bind_cols(feats)
}

# Run
features_df <- map_dfr(seq_len(nlyr(r)), process_one_layer)