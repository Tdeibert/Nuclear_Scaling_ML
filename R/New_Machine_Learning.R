# ---- packages ----
library(tidyverse)
library(terra)        # raster I/O
library(EBImage)      # image ops, connected components
library(h2o)          # ML (unsupervised/supervised)
library(clue)         # Hungarian assignment for tracking
library(glue)
library(slider)

# ---- utilities: array <-> tibble helpers ----
as_ebimg <- function(mat) {
  # EBImage expects numeric matrix in [0,1]
  rng <- range(mat, na.rm = TRUE)
  if (diff(rng) == 0) return Image(array(0, dim = dim(mat)))
  Image((mat - rng[1]) / diff(rng))
}

stack_to_array <- function(r) {
  # terra SpatRaster -> 3D array [y, x, layers]
  v <- as.array(r) # returns [layers, y, x]
  aperm(v, c(2,3,1))
}

# ---- feature engineering for pixel classification ----
make_pixel_features <- function(img2d, scales = c(1, 2, 4)) {
  # img2d: numeric matrix (nrow = y, ncol = x), nuclei channel
  im <- as_ebimg(img2d)
  
  # base features
  base_int <- im
  grad_mag <- sqrt(gfilter(im, c(1,0))^2 + gfilter(im, c(0,1))^2)
  
  # multi-scale LoG/DoG-ish responses
  feats <- list(
    intensity = base_int,
    grad = grad_mag
  )
  for (s in scales) {
    blur <- gblur(im, sigma = s)
    lap  <- filter2(blur, makeBrush(3, shape="diamond")) - blur # cheap high-pass
    feats[[paste0("blur_", s)]] <- blur
    feats[[paste0("lap_", s)]]  <- lap
  }
  
  arrs <- lapply(feats, function(f) as.array(f))
  H <- nrow(arrs[[1]]); W <- ncol(arrs[[1]])
  df <- tibble(
    y = rep(seq_len(H), times = W),
    x = rep(seq_len(W), each = H)
  )
  for (nm in names(arrs)) {
    df[[nm]] <- as.vector(arrs[[nm]])
  }
  df
}

# ---- unsupervised nuclei segmentation via H2O K-means ----
segment_nuclei_unsup <- function(feature_df, k = 3, nuclei_rule = c("brightest_cluster")) {
  # Start/attach H2O once per session
  if (!h2o::h2o.connectionIsUp()) h2o::h2o.init(nthreads = -1)
  # Select numeric feature columns
  feat_cols <- setdiff(names(feature_df), c("x","y","t","z"))
  hf <- as.h2o(feature_df[, feat_cols, drop = FALSE])
  km <- h2o::h2o.kmeans(training_frame = hf, k = k, standardize = TRUE, seed = 1)
  cl  <- as.vector(h2o::h2o.predict(km, hf)$predict)
  
  # Heuristic: choose the cluster with highest mean intensity as "nucleus" class
  means <- feature_df %>%
    mutate(.cl = cl) %>%
    group_by(.cl) %>%
    summarize(mu = mean(intensity, na.rm = TRUE), .groups="drop") %>%
    arrange(desc(mu))
  
  nucleus_cluster <- means$.cl[1]
  mask <- as.integer(cl == nucleus_cluster)
  feature_df$mask <- mask
  feature_df
}

# ---- 2D labeling -> nuclei centroids for one plane ----
label_nuclei_2d <- function(mask_df, H, W, min_area = 20) {
  m <- matrix(mask_df$mask, nrow = H, ncol = W, byrow = FALSE)
  im <- Image(m)
  im <- opening(im, makeBrush(3, shape = "disc"))
  im <- fillHull(im)
  lab <- bwlabel(im)
  props <- computeFeatures.moment(lab) %>% as_tibble(.name_repair="minimal")
  bsize <- computeFeatures.shape(lab) %>% as_tibble(.name_repair="minimal")
  
  keep_ids <- which(bsize$s.area >= min_area)
  if (length(keep_ids) == 0) return(tibble())
  
  props %>%
    mutate(label = row_number()) %>%
    filter(label %in% keep_ids) %>%
    transmute(
      label,
      x = m.cx,
      y = m.cy,
      area = bsize$s.area[keep_ids]
    )
}

# ---- droplet detection (circle-ish) from a droplet channel ----
# Approach: binarize, label droplets, compute equivalent diameter
detect_droplets_2d <- function(droplet_img2d, min_diam_px = 10) {
  im <- as_ebimg(droplet_img2d)
  thr <- otsu(im)
  bw  <- im > thr
  bw  <- opening(bw, makeBrush(5, shape="disc"))
  bw  <- fillHull(bw)
  lab <- bwlabel(bw)
  m   <- computeFeatures.moment(lab) %>% as_tibble()
  s   <- computeFeatures.shape(lab)  %>% as_tibble()
  
  if (nrow(m) == 0) return(tibble())
  
  out <- tibble(
    droplet_id = seq_len(nrow(m)),
    x = m$m.cx,
    y = m$m.cy,
    area_px = s$s.area,
    equiv_diam_px = 2*sqrt(s$s.area/pi)
  ) %>% filter(equiv_diam_px >= min_diam_px)
  
  list(table = out, label_img = lab)
}

# ---- assign nuclei to droplets & fetch diameter ----
assign_nuclei_to_droplets <- function(nuclei_xy, droplet_lab) {
  if (nrow(nuclei_xy) == 0) return(nuclei_xy %>% mutate(droplet_id = NA_integer_, droplet_diam_px = NA_real_))
  coords <- round(cbind(nuclei_xy$y, nuclei_xy$x)) # EBImage uses [row,col] = [y,x]
  coords[,1] <- pmax(pmin(coords[,1], nrow(droplet_lab)), 1)
  coords[,2] <- pmax(pmin(coords[,2], ncol(droplet_lab)), 1)
  droplet_ids <- droplet_lab[cbind(coords[,1], coords[,2])]
  nuclei_xy$droplet_id <- as.integer(droplet_ids)
  nuclei_xy
}

# ---- simple frame-to-frame tracking (Hungarian) ----
track_nuclei <- function(df, max_disp = 20) {
  # df: nuclei positions with columns (t, id_local, x, y)
  # Returns: track_id stable across frames
  frames <- sort(unique(df$t))
  next_track <- 1L
  df$track_id <- NA_integer_
  
  # initialize with first frame
  f0 <- frames[1]
  idx0 <- which(df$t == f0)
  df$track_id[idx0] <- seq(next_track, length.out = length(idx0))
  next_track <- next_track + length(idx0)
  
  for (k in 2:length(frames)) {
    f_prev <- frames[k-1]; f_cur <- frames[k]
    A <- df %>% filter(t == f_prev)
    B <- df %>% filter(t == f_cur)
    if (nrow(A) == 0 || nrow(B) == 0) next
    
    # cost = distance matrix
    cost <- as.matrix(dist(rbind(A[,c("x","y")], B[,c("x","y")])))
    cost <- cost[seq_len(nrow(A)), nrow(A)+seq_len(nrow(B))]
    # large cost for > max_disp
    cost[cost > max_disp] <- 1e6
    
    assign <- clue::solve_LSAP(cost)  # column index each row mapped to
    mapped_j <- as.integer(assign)
    
    # set track IDs where feasible
    for (i in seq_len(nrow(A))) {
      j <- mapped_j[i]
      if (j >= 1 && j <= nrow(B) && is.finite(cost[i,j]) && cost[i,j] < 1e6) {
        # inherit track
        B_idx <- which(df$t == f_cur)[j]
        A_idx <- which(df$t == f_prev)[i]
        df$track_id[B_idx] <- df$track_id[A_idx]
      }
    }
    # new/unmatched in B -> new tracks
    unmatched_B <- which(is.na(df$track_id[df$t == f_cur]))
    if (length(unmatched_B)) {
      df$track_id[which(df$t == f_cur)[unmatched_B]] <- seq(next_track, length.out = length(unmatched_B))
      next_track <- next_track + length(unmatched_B)
    }
  }
  df
}

# ---- main entry: per-timepoint nuclei on MIP across Z, plus droplet diameter ----
# Assumes a multi-layer raster where you can index channels/Z/time externally.
# You can feed already-sliced matrices to keep this function focused.
detect_nuclei_in_field <- function(
    nuclei_stack_list,     # list over time: each is 3D array [y,x,z] of the nuclei channel
    droplet_stack_list,    # list over time: each is 2D matrix [y,x] (droplet channel MIP or phase)
    px_size_um = 0.1,      # pixel size for diameter conversion
    min_nucleus_area_px = 20,
    k_clusters = 3
) {
  if (length(nuclei_stack_list) != length(droplet_stack_list))
    stop("nuclei_stack_list and droplet_stack_list must be same length (per time).")
  
  # In case H2O not up yet:
  if (!h2o::h2o.connectionIsUp()) h2o::h2o.init(nthreads = -1)
  
  out_list <- vector("list", length(nuclei_stack_list))
  
  for (tt in seq_along(nuclei_stack_list)) {
    nuc_3d <- nuclei_stack_list[[tt]]      # [y,x,z]
    # MIP across Z for detection; you can replace with focus-based slice selection later
    nuc_mip <- apply(nuc_3d, c(1,2), max, na.rm = TRUE)
    
    # Features -> unsupervised segmentation
    feats <- make_pixel_features(nuc_mip)
    H <- nrow(nuc_mip); W <- ncol(nuc_mip)
    seg  <- segment_nuclei_unsup(feats, k = k_clusters)
    
    nuclei_xy <- label_nuclei_2d(seg, H, W, min_area = min_nucleus_area_px)
    if (nrow(nuclei_xy) == 0) {
      out_list[[tt]] <- tibble(t = tt, id_local = integer(), x = double(), y = double(),
                               area_px = double(), droplet_id = integer(), droplet_diam_px = double())
      next
    }
    
    # droplet detection for this time
    drop_2d <- droplet_stack_list[[tt]]
    droplet_det <- detect_droplets_2d(drop_2d)
    droplet_tbl <- droplet_det$table
    droplet_lab <- droplet_det$label_img
    
    # assign nuclei -> droplets
    nuclei_xy2 <- assign_nuclei_to_droplets(nuclei_xy, droplet_lab)
    nuclei_xy2 <- nuclei_xy2 %>%
      left_join(droplet_tbl %>% select(droplet_id, equiv_diam_px), by = "droplet_id") %>%
      rename(droplet_diam_px = equiv_diam_px)
    
    out_list[[tt]] <- nuclei_xy2 %>%
      transmute(
        t = tt,
        id_local = label,
        x, y,
        area_px = area,
        droplet_id,
        droplet_diam_px
      )
  }
  
  nuclei_all <- bind_rows(out_list)
  
  # tracking across time (simple NN/Hungarian)
  nuclei_tracks <- track_nuclei(nuclei_all, max_disp = 20)
  
  # add microns for droplet diameter
  nuclei_tracks <- nuclei_tracks %>%
    mutate(
      droplet_diam_um = droplet_diam_px * px_size_um
    ) %>%
    arrange(t, track_id)
  
  list(
    nuclei = nuclei_tracks,
    params = list(px_size_um = px_size_um, min_nucleus_area_px = min_nucleus_area_px, k_clusters = k_clusters)
  )
}

# ---- Example usage (pseudo; plug your I/O) ----
# 1) Load your stacks with terra or your existing loader, slice per time:
# r <- terra::rast("path/to/stack.tif")
# ... split into nuclei_stack_list[[t]]: array [y,x,z] for the nuclei channel
# ... and droplet_stack_list[[t]]: matrix [y,x] (droplet channel or phase MIP)
#
# 2) Run:
# res <- detect_nuclei_in_field(nuclei_stack_list, droplet_stack_list, px_size_um = 0.108)
# head(res$nuclei)
