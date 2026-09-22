#### Robust Machine learning Pipeline for image Analysis#### 

# ---- packages ----
library(EBImage)
library(tidyverse)

# ---- config (edit here) ----
cfg <- list(
  # image I/O
  path = "path/to/your/frame.png",    # e.g., "e34caf3d-22d4-4fba-82bc-4f4240ff260b.png"
  # pixel geometry
  px_size_um = 0.325,                 # <-- set your pixel size (µm/px)
  # droplet size constraints (µm diameter)
  droplet_diam_um_min = 40,
  droplet_diam_um_max = 55,
  # nucleus area constraints (µm^2)
  nucleus_area_um2_min = 40,
  nucleus_area_um2_max = 500,
  # preprocessing & thresholds
  smooth_sigma = 1.0,                 # Gaussian blur sigma (px)
  thr_w = 51, thr_h = 51,             # adaptive thresh window (odd px)
  thr_offset = 0.02,                  # smaller = more foreground
  # seeds & morphology
  erode_brush = 7,                    # seed shrinking (px; disc radius)
  open_brush  = 3,                    # small opening to clean specks
  # nucleus enhancement
  nuc_tophat_radius = 5,              # px (structuring element for top-hat)
  nuc_thr_w = 21, nuc_thr_h = 21,     # local threshold for nuclei
  nuc_thr_offset = 0.02,
  # QC overlay
  out_prefix = "demo_droplet_nuclei"
)

# ---- helpers ----
px2um <- function(px) px * cfg$px_size_um
um2px <- function(um) um / cfg$px_size_um

area_px_to_diam_um <- function(area_px) {
  diam_px <- 2 * sqrt(area_px / pi)
  px2um(diam_px)
}

# ---- load image (single channel) ----
img <- readImage(cfg$path) %>% normalize()
if (colorMode(img) != 0) img <- channel(img, "gray")

# ---- preprocess ----
img_s <- gblur(img, sigma = cfg$smooth_sigma)             # mild denoise
# adaptive threshold to get droplet interiors as foreground
bin0  <- thresh(img_s, w = cfg$thr_w, h = cfg$thr_h, offset = cfg$thr_offset)
bin0  <- opening(bin0, makeBrush(cfg$open_brush, "disc"))

# ---- seeds + watershed-like via propagate on -distmap ----
D      <- distmap(bin0)
seeds  <- erode(bin0, makeBrush(cfg$erode_brush, "disc"))
seedsL <- bwlabel(seeds)
# grow seeds inside foreground, using -distance as metric → boundaries stop at narrow gaps
dropL  <- propagate(-D, seeds = seedsL, mask = bin0)
dropL  <- bwlabel(dropL > 0)  # relabel compactly

# ---- droplet feature table & size filtering ----
drop_feat <- as_tibble(computeFeatures.shape(dropL)) %>%
  mutate(
    droplet_id = row_number(),
    diam_um = area_px_to_diam_um(s.area),
    keep_size = between(diam_um, cfg$droplet_diam_um_min, cfg$droplet_diam_um_max)
  )

keep_ids <- drop_feat %>% filter(keep_size) %>% pull(droplet_id)
dropL_filt <- dropL
dropL_filt[!dropL_filt %in% keep_ids] <- 0L
dropL_filt <- bwlabel(dropL_filt > 0)  # compact relabel after filtering

# recompute features after filtering (IDs changed)
drop_feat <- as_tibble(computeFeatures.shape(dropL_filt)) %>%
  mutate(droplet_id = row_number(),
         diam_um = area_px_to_diam_um(s.area))

# droplet centroids for later joins/overlays
drop_mom <- as_tibble(computeFeatures.moment(dropL_filt)) %>%
  mutate(droplet_id = row_number(),
         cx = m.cx, cy = m.cy)

droplets_tbl <- drop_feat %>%
  select(droplet_id, area_px = s.area, perimeter_px = s.perimeter, diam_um) %>%
  left_join(drop_mom %>% select(droplet_id, cx, cy), by = "droplet_id")

# ---- nuclei detection (inside droplets) ----
# enhance spots with white top-hat: image - opening(image)
se_radius <- cfg$nuc_tophat_radius
top_hat   <- img_s - opening(img_s, makeBrush(se_radius, "disc"))
nuc_bin0  <- thresh(top_hat, w = cfg$nuc_thr_w, h = cfg$nuc_thr_h, offset = cfg$nuc_thr_offset)
nuc_bin0  <- opening(nuc_bin0, makeBrush(3, "disc"))

# keep only nuclei inside kept droplets
# (mask by droplet foreground)
droplet_mask <- dropL_filt > 0
nuc_bin <- nuc_bin0 & droplet_mask

# label nuclei and measure
nucL <- bwlabel(nuc_bin)
if (max(nucL) > 0) {
  nuc_shape <- as_tibble(computeFeatures.shape(nucL)) %>%
    mutate(nuc_id = row_number())
  nuc_mom   <- as_tibble(computeFeatures.moment(nucL)) %>%
    mutate(nuc_id = row_number(),
           cx = m.cx, cy = m.cy)
  nuc_tbl <- nuc_shape %>%
    transmute(nuc_id, area_px = s.area) %>%
    left_join(nuc_mom %>% select(nuc_id, cx, cy), by = "nuc_id") %>%
    mutate(area_um2 = area_px * (cfg$px_size_um^2),
           keep_area = between(area_um2, cfg$nucleus_area_um2_min, cfg$nucleus_area_um2_max))
  
  # drop nuclei outside area range
  keep_nuc_ids <- nuc_tbl %>% filter(keep_area) %>% pull(nuc_id)
  nucL_filt <- nucL
  nucL_filt[!nucL_filt %in% keep_nuc_ids] <- 0L
  nucL_filt <- bwlabel(nucL_filt > 0)
  
  # recompute nuclei features after filtering
  nuc_shape <- as_tibble(computeFeatures.shape(nucL_filt)) %>%
    mutate(nuc_id = row_number())
  nuc_mom   <- as_tibble(computeFeatures.moment(nucL_filt)) %>%
    mutate(nuc_id = row_number(), cx = m.cx, cy = m.cy)
  nuc_tbl <- nuc_shape %>%
    transmute(nuc_id, area_px = s.area) %>%
    left_join(nuc_mom %>% select(nuc_id, cx, cy), by = "nuc_id") %>%
    mutate(area_um2 = area_px * (cfg$px_size_um^2))
  
  # ---- assign each nucleus to its enclosing droplet (by centroid lookup) ----
  # EBImage indexes [row, col] == [y, x]
  idx <- cbind(pmin(pmax(round(nuc_tbl$cy), 1), dim(dropL_filt)[1]),
               pmin(pmax(round(nuc_tbl$cx), 1), dim(dropL_filt)[2]))
  nuc_tbl$droplet_id <- dropL_filt[idx]
  nuc_tbl <- nuc_tbl %>% filter(droplet_id > 0)
  
} else {
  nuc_tbl <- tibble(nuc_id = integer(), area_px = integer(), cx = double(),
                    cy = double(), area_um2 = double(), droplet_id = integer())
  nucL_filt <- nucL
}

# ---- per-droplet nucleus counts & exclusion flags ----
counts_tbl <- nuc_tbl %>%
  count(droplet_id, name = "nuc_count")

droplets_tbl <- droplets_tbl %>%
  left_join(counts_tbl, by = "droplet_id") %>%
  mutate(nuc_count = replace_na(nuc_count, 0L),
         excluded_multi_nuc = nuc_count > 1L)

# ---- save tables ----
write_csv(droplets_tbl, paste0(cfg$out_prefix, "_droplets.csv"))
write_csv(nuc_tbl,      paste0(cfg$out_prefix, "_nuclei.csv"))

# ---- QC overlays ----
# 1) Droplet boundaries (kept) on background
overlay1 <- paintObjects(dropL_filt, toRGB(img), col = "#00FF00") # green edges
# 2) Mark multi-nucleus droplets in red by painting their regions
multi_mask <- dropL_filt %in% (droplets_tbl %>% filter(excluded_multi_nuc) %>% pull(droplet_id))
overlay2 <- paintObjects(multi_mask, overlay1, col = "#FF0000")   # red fill (semi)
# 3) Paint nuclei (white)
overlay3 <- paintObjects(nucL_filt, overlay2, col = "#FFFFFF")

writeImage(overlay3, paste0(cfg$out_prefix, "_overlay.png"))

# ---- console summary ----
cat("\n=== SUMMARY ===\n")
cat("Droplets (kept by size):", nrow(droplets_tbl), "\n")
cat("  - with 0 nuclei:", sum(droplets_tbl$nuc_count == 0), "\n")
cat("  - with 1 nucleus:", sum(droplets_tbl$nuc_count == 1), "\n")
cat("  - with >1 nuclei (excluded):", sum(droplets_tbl$excluded_multi_nuc), "\n")
cat("Nuclei (kept by area):", nrow(nuc_tbl), "\n")
cat("Overlay written to:", paste0(cfg$out_prefix, "_overlay.png"), "\n")
cat("Tables written to  :", paste0(cfg$out_prefix, "_droplets.csv / _nuclei.csv"), "\n")
