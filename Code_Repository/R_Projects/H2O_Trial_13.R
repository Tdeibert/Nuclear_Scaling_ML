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
#sigma is now the intensity value for background for the channel slice. 
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

#### 4. Apply Otsu threshold ####
otsu_thresholds <- r_df_labeled %>%
  group_by(channel, z, time) %>%
  summarise(
    otsu_thresh = calculate_otsu_threshold(intensity),
    .groups = "drop"
  )

r_above_bg <- r_df_labeled %>%
  left_join(otsu_thresholds, by = c("channel", "z", "time")) %>%
  filter(intensity > otsu_thresh)


#### 5. Per-layer K-means clustering ####
k_clusters <- 1 

r_clustered <- r_above_bg %>%
  group_by(channel, z, time) %>%
  group_modify(~ {
    h2o_frame <- as.h2o(.x %>% select(x, y))
    km_model  <- h2o.kmeans(h2o_frame, k = k_clusters, standardize = TRUE, seed = 123)
    pred      <- h2o.predict(km_model, h2o_frame) %>% as.data.frame()
    .x %>% mutate(cluster = pred$predict)
  }) %>%
  ungroup()




#### 6. Convert to sf and rasterize clustered pixel intensities ####
intensity_raster <- r_clustered %>%
  select(x, y, intensity) %>%
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  rasterize(img[[1]], field = "intensity")

#Making a cluster Raster 
cluster_raster <- r_clustered %>%
  select(x, y, cluster) %>% 
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  rasterize(img[[1]], field = "cluster")

#### 7. Plotting Rasters #### 
# Choose the channel
channel_to_plot <- 2

# Filter for selected channel
channel_data <- r_clustered %>%
  filter(channel == channel_to_plot)

# Set up plotting grid
n_z <- max(channel_data$z)
plot_cols <- ceiling(sqrt(n_z)) #why sqrt? I am unsure why It needed to add the math here? 
plot_rows <- ceiling(n_z / plot_cols)
par(mfrow = c(plot_rows, plot_cols), mar = c(1, 1, 2, 1))  # adjust margins

# Loop over Z slices
for (z_slice in sort(unique(channel_data$z))) {
  layer_data <- channel_data %>%
    filter(z == z_slice) %>%
    select(x, y, intensity) %>%
    drop_na()
  
  if (nrow(layer_data) > 0) {
    layer_raster <- layer_data %>%
      st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
      rasterize(img[[1]], field = "intensity")
    
    plot(layer_raster, main = glue("Channel {channel_to_plot}, Z {z_slice}"))
  } else {
    plot.new()
    title(main = glue("Channel {channel_to_plot}, Z {z_slice}\n(no data)"))
  }
}

####8. Making Masks Based on Clusters #### 
testr<- r_df_labeled %>% 
  filter(layer_name == "T1_Z8_C2") %>%
  left_join(r_clustered %>% 
              filter(layer_name == "T1_Z8_C2"), by = c("x", "y", "layer_name")) %>% 
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  vect() %>% 
  rasterize(y = img[[1]], field = "cluster")

testp<-testr %>% 
  as.polygons() %>% 
  st_as_sf() %>% 
  st_cast("POLYGON") %>% 
  mutate(area = st_area(geometry),
         polyid = 1:n()) %>% 
  filter(area > 10)

test<- mask(img[[22:24]], testp)
global(test, "sum", na.rm = TRUE)
plot(test)


#binary foreground mask (1 = clustered object)
r_binary_mask_data <- r_clustered %>%
  mutate(mask_value = 1) %>%
  select(x, y, z, time, channel, mask_value)

#### 9. Plot binary masks for each Z-layer, time, and channel ####
channels_to_plot <- sort(unique(r_binary_mask_data$channel))
n_z <- max(r_binary_mask_data$z)
n_time <- max(r_binary_mask_data$time)

for (ch in channels_to_plot) {
  for (t in 1:n_time) {
    cat(glue::glue("Plotting Binary Mask - Channel {ch}, Time {t}\n"))
    
    ch_time_data <- r_binary_mask_data %>%
      filter(channel == ch, time == t)
    
    plot_cols <- ceiling(sqrt(n_z)) #why did it add this sqrt function? 
    plot_rows <- ceiling(n_z / plot_cols)
    par(mfrow = c(plot_rows, plot_cols), mar = c(1, 1, 2, 1))
    
    for (z_slice in sort(unique(ch_time_data$z))) {
      layer_data <- ch_time_data %>%
        filter(z == z_slice)
      
      if (nrow(layer_data) > 0) {
        binary_raster <- layer_data %>%
          st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
          rasterize(img[[1]], field = "mask_value")
        
        binary_raster[is.na(binary_raster)] <- 0
        
        plot(binary_raster,
             main = glue("C{ch} T{t} Z{z_slice}"),
             col = c("black", "white"))
      } else {
        plot.new()
        title(main = glue("C{ch} T{t} Z{z_slice} (no data)"))
      }
    }
  }
}


#model_auto_k <- h2o.kmeans(
  #training_frame = h2o_data,
  #k = 20,                    # upper limit
  #estimate_k = TRUE,        # let H2O choose
  #seed = 123
#)

#model_auto_k@model$model_summary

unique(r_clustered$cluster)
