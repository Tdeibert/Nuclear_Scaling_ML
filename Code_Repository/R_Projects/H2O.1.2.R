library(terra)
library(h2o)
library(tidyverse)
library(sf)
library(glue)

# Start H2O
h2o.init(max_mem_size = "8G")

#### 1. Load and structure raster stack ####
raster_path <- "./Test_Stack.tif"
img <- rast(raster_path)

n_channels <- 3
n_z        <- 20
n_time     <- 2
expected_layers <- n_channels * n_z * n_time
actual_layers   <- nlyr(img)

if (actual_layers != expected_layers) {
  stop(glue::glue("Mismatch in layers: expected {expected_layers}, got {actual_layers}"))
}

layer_info <- expand_grid(
  time    = 1:n_time,
  z       = 1:n_z,
  channel = 1:n_channels
) %>%
  mutate(layer_index = row_number(),
         layer_name  = str_glue("T{time}_Z{z}_C{channel}"))

names(img) <- layer_info$layer_name

#### 2. Convert to long format dataframe ####
r_df_labeled <- as.data.frame(img, xy = TRUE, cells = FALSE, na.rm = FALSE) %>%
  as_tibble() %>%
  pivot_longer(cols = -c(x, y), names_to = "layer_name", values_to = "intensity") %>%
  left_join(layer_info, by = "layer_name")

#### 3. Compute Otsu threshold per (channel, z, time) ####
calculate_otsu_threshold <- function(intensity_values, nbins = 256) {
  h <- hist(intensity_values, breaks = nbins, plot = FALSE)
  p <- h$counts / sum(h$counts)
  omega <- cumsum(p)
  mu <- cumsum(p * h$mids)
  mu_t <- mu[length(mu)]
  sigma_b_squared <- (mu_t * omega - mu)^2 / (omega * (1 - omega))
  sigma_b_squared[is.nan(sigma_b_squared)] <- 0
  threshold <- h$mids[which.max(sigma_b_squared)] 
  return(threshold)
}

otsu_thresholds <- r_df_labeled %>%
  group_by(channel, z, time) %>%
  summarise(
    otsu_thresh = calculate_otsu_threshold(intensity),
    .groups = "drop"
  )

r_above_bg <- r_df_labeled %>%
  left_join(otsu_thresholds, by = c("channel", "z", "time")) %>%
  filter(intensity > otsu_thresh)

#### 4. Auto-select K with elbow method ####
get_best_k <- function(df, max_k = 10, threshold = 0.1) {
  if (nrow(df) < 5) return(1)  # Not enough points to cluster
  
  h2o_frame <- as.h2o(df %>% select(x, y))
  
  wss_list <- map(1:max_k, function(k) {
    tot_wss <- tryCatch({
      model <- h2o.kmeans(h2o_frame, k = k, standardize = TRUE, seed = 123)
      model@model$tot_withinss
    }, error = function(e) {
      message(glue("K = {k} failed: {e$message}"))
      NA_real_
    })
    
    tibble(k = k, tot_withinss = tot_wss)
  })
  
  wss_values <- bind_rows(wss_list)
  
  if (!"tot_withinss" %in% names(wss_values)) {
    message("No valid clustering results — defaulting to k = 1")
    return(1)
  }
  
  wss_values <- wss_values %>%
    filter(!is.na(tot_withinss)) %>%
    arrange(k) %>%
    mutate(
      wss_drop = lag(tot_withinss) - tot_withinss,
      rel_drop = wss_drop / lag(tot_withinss)
    )
  
  best_k <- wss_values %>%
    filter(rel_drop < threshold) %>%
    slice(1) %>%
    pull(k)
  
  if (length(best_k) == 0 || is.na(best_k)) best_k <- max_k
  return(best_k)
}

r_clustered <- r_above_bg %>%
  group_by(channel, z, time) %>%
  group_modify(~ {
    if (nrow(.x) < 5) {
      .x$cluster <- NA
      return(.x)
    }
    
    best_k    <- get_best_k(.x, max_k = 10, threshold = 0.1)
    h2o_frame <- as.h2o(.x %>% select(x, y))
    km_model  <- h2o.kmeans(h2o_frame, k = best_k, standardize = TRUE, seed = 123)
    pred      <- h2o.predict(km_model, h2o_frame) %>% as.data.frame()
    .x %>% mutate(cluster = pred$predict)
  }) %>%
  ungroup()

#### 5. Rasterization ####
intensity_raster <- r_clustered %>%
  select(x, y, intensity) %>%
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  rasterize(img[[1]], field = "intensity")

cluster_raster <- r_clustered %>%
  select(x, y, cluster) %>% 
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  rasterize(img[[1]], field = "cluster")

#### 6. Optional Visualization ####
channel_to_plot <- 2
channel_data <- r_clustered %>% filter(channel == channel_to_plot)
n_z <- max(channel_data$z)
plot_cols <- ceiling(sqrt(n_z))
plot_rows <- ceiling(n_z / plot_cols)
par(mfrow = c(plot_rows, plot_cols), mar = c(1, 1, 2, 1))

for (z_slice in sort(unique(channel_data$z))) {
  layer_data <- channel_data %>%
    filter(z == z_slice) %>%
    select(x, y, cluster) %>%
    drop_na()
  if (nrow(layer_data) > 0) {
    layer_raster <- layer_data %>%
      st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
      rasterize(img[[1]], field = "cluster")
    plot(layer_raster, main = glue("Channel {channel_to_plot}, Z {z_slice}"))
  } else {
    plot.new()
    title(main = glue("Channel {channel_to_plot}, Z {z_slice}\n(no data)"))
  }
}

#### 7. Create polygon masks from clusters ####
testr <- r_df_labeled %>% 
  filter(layer_name == "T1_Z8_C2") %>%
  left_join(r_clustered %>% filter(layer_name == "T1_Z8_C2"), 
            by = c("x", "y", "layer_name")) %>% 
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  vect() %>% 
  rasterize(y = img[[1]], field = "cluster")

testp <- testr %>% 
  as.polygons() %>% 
  st_as_sf() %>% 
  st_cast("POLYGON") %>% 
  mutate(area = st_area(geometry),
         polyid = 1:n()) %>% 
  filter(area > 10)

test <- mask(img[[22:24]], testp)
global(test, "sum", na.rm = TRUE)
plot(test)
unique(r_clustered$cluster)
