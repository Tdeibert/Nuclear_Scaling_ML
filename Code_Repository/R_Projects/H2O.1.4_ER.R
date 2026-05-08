library(terra)
library(h2o)
library(tidyverse)
library(sf)
library(glue)
library(FNN)
# Start H2O
h2o.init(max_mem_size = "8G")

#### 1. Load and structure raster stack ####
raster_path <- "./Test_Stack.tif"
img <- rast(raster_path)
plot(img, 20:30)

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

#### 6. Compute nucleus centroids from clusters in channel 2 ####
nucleus_clusters <- r_clustered %>%
  filter(channel == 2) %>%
  drop_na(cluster)

nucleus_centroids <- nucleus_clusters %>%
  group_by(z, time, cluster) %>%
  summarise(
    x_nuc = mean(x),
    y_nuc = mean(y),
    .groups = "drop"
  ) %>%
  rename(nucleus_id = cluster)

#### 7. Assign ER pixels to nearest nucleus centroid ####
er_pixels <- r_clustered %>%
  filter(channel == 1) %>%
  select(x, y, z, time, intensity)

assign_nearest_nucleus <- function(er_df, nuc_df) {
  er_df %>%
    group_by(z, time) %>%
    group_modify(~ {
      nuc <- filter(nuc_df, z == .y$z, time == .y$time)
      if (nrow(nuc) == 0) return(.x %>% mutate(nucleus_id = NA, dist_to_nucleus = NA))
      nn <- get.knnx(nuc[, c("x_nuc", "y_nuc")], .x[, c("x", "y")], k = 1)
      .x %>%
        mutate(
          nucleus_id = nuc$nucleus_id[nn$nn.index[, 1]],
          dist_to_nucleus = nn$nn.dist[, 1]
        )
    }) %>%
    ungroup()
}

er_with_nuclei <- assign_nearest_nucleus(er_pixels, nucleus_centroids)

#### 8. Compute ER organization features per nucleus ####
er_features <- er_with_nuclei %>%
  group_by(z, time, nucleus_id) %>%
  summarise(
    compaction_score = mean(intensity, na.rm = TRUE),
    membrane_total   = sum(intensity, na.rm = TRUE),
    distance_mean    = mean(dist_to_nucleus, na.rm = TRUE),
    .groups = "drop"
  )

er_features_h2o <- as.h2o(er_features %>% select(-nucleus_id))

km_model <- h2o.kmeans(
  training_frame = er_features_h2o,
  k = 4,
  standardize = TRUE,
  seed = 123
)

h2o.centroid_stats(km_model)

er_features$cluster <- as.vector(h2o.predict(km_model, er_features_h2o)$predict)


#adding in a new H20 model that will run for each Z and will predict the correct number of K values. 
er_clustered <- er_with_nuclei %>%
  group_by(z) %>%  # optionally add time if time-varying ER behavior is expected
  group_modify(~ {
    if (nrow(.x) < 10) return(.x %>% mutate(er_cluster = NA))  # skip sparse slices
    
    h2o_frame <- as.h2o(.x %>% select(x, y))
    
    km_model <- h2o.kmeans(
      training_frame = h2o_frame,
      k = 10,  # upper bound
      estimate_k = TRUE,
      standardize = TRUE,
      seed = 123
    )
    
    pred <- h2o.predict(km_model, h2o_frame) %>% as.data.frame()
    .x %>% mutate(er_cluster = pred$predict)
  }) %>%
  ungroup()

er_clustered <- er_clustered %>%
  mutate(er_cluster = as.factor(er_cluster))


ggplot(filter(er_clustered, time == 1), aes(x = x, y = y, fill = er_cluster)) +
  geom_raster() +
  coord_equal() +
  scale_fill_viridis_d(na.value = "black") +
  facet_wrap(~ z, ncol = 5) +
  labs(title = "ER Clusters by Z (Time 1)", fill = "ER Cluster") +
  theme_minimal()
# Merge ER cluster results back
er_with_cluster <- er_with_nuclei %>%
  left_join(er_features %>% select(z, time, nucleus_id, cluster), by = c("z", "time", "nucleus_id"))




# Prepare cluster-labeled ER pixels
cluster_plot_data <- er_with_cluster %>%
  drop_na(cluster) %>%
  mutate(cluster = as.factor(cluster))  # for color mapping

#### Plot faceted ER cluster rasters with distinct colors per cluster ID####
cluster_ids_all <- levels(cluster_plot_data$cluster)
cluster_palette <- setNames(viridis::viridis(length(cluster_ids_all)), cluster_ids_all)

n_z <- max(cluster_plot_data$z)
n_time <- max(cluster_plot_data$time)
plot_cols <- ceiling(sqrt(n_z))
plot_rows <- ceiling(n_z / plot_cols)

for (t in 1:n_time) {
  cat(glue("Plotting ER Clusters - Time {t}\n"))
  par(mfrow = c(plot_rows, plot_cols), mar = c(1, 1, 2, 1))
  
  for (z_slice in 1:n_z) {
    layer_data <- cluster_plot_data %>%
      filter(z == z_slice, time == t)
    
    if (nrow(layer_data) > 0) {
      cluster_rast <- layer_data %>%
        mutate(cluster = as.integer(as.character(cluster))) %>%
        st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
        rasterize(img[[1]], field = "cluster")
      
      cluster_vals <- sort(as.numeric(as.character(na.omit(unique(layer_data$cluster)))))
      cluster_colors <- cluster_palette[as.character(cluster_vals)]
      
      plot(cluster_rast,
           main = glue("Time {t}, Z {z_slice}"),
           col = cluster_colors,
           breaks = c(cluster_vals - 0.5, max(cluster_vals) + 0.5),
           legend = FALSE)
    } else {
      plot.new()
      title(main = glue("Time {t}, Z {z_slice}\n(no data)"))
    }
  }
  
  # Place legend after each timepoint
  par(mfrow = c(1, 1))
  plot.new()
  legend("center",
         legend = cluster_ids_all,
         fill = cluster_palette,
         title = "Cluster ID",
         cex = 1.2,
         bty = "n")
}

unique(cluster_plot_data$cluster)


####Making a cluster Raster#### 
cluster_raster <- r_clustered %>%
  select(x, y, cluster) %>% 
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = crs(img)) %>%
  rasterize(img[[1]], field = "cluster")
plot(cluster_raster)

####GGPlots####
# Summary of ER clusters Without Filtering for neighboring droplets 
er_cluster_summary <- cluster_plot_data %>%
  group_by(time, z, nucleus_id, cluster) %>%
  summarise(
    mean_distance       = mean(dist_to_nucleus, na.rm = TRUE),
    total_intensity     = sum(intensity, na.rm = TRUE),
    area_pixels         = n(),  # number of pixels in cluster
    intensity_density   = total_intensity / area_pixels,
    cluster_centroid_x  = mean(x),
    cluster_centroid_y  = mean(y),
    .groups = "drop"
  )

#with filtering for other droplets 
er_cluster_summary <- er_clustered_filtered %>%
  group_by(time, z, nucleus_id, er_cluster) %>%
  summarise(
    mean_distance     = mean(dist_to_nucleus, na.rm = TRUE),
    total_intensity   = sum(intensity, na.rm = TRUE),
    area_pixels       = n(),
    intensity_density = total_intensity / area_pixels,
    cluster_centroid_x = mean(x),
    cluster_centroid_y = mean(y),
    .groups = "drop"
  )

#IntDen Summary of each cluster box plot
ggplot(er_cluster_summary, aes(x = factor(cluster), y = intensity_density, fill = factor(cluster))) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  labs(x = "Cluster", y = "Intensity per Pixel (Density)", fill = "Cluster") +
  theme_minimal()
#Mean Distance of each cluster identitity from the nucleus centroid
ggplot(er_cluster_summary, aes(x = factor(cluster), y = mean_distance, fill = factor(cluster))) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  labs(x = "Cluster ID", y = "Mean Distance to Nucleus", fill = "Cluster") +
  theme_minimal()

#Total Intensity of the ER cluster
ggplot(er_cluster_summary, aes(x = factor(cluster), y = total_intensity, fill = factor(cluster))) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  labs(x = "Cluster ID", y = "Total ER Intensity", fill = "Cluster") +
  theme_minimal()

#Distance of eeach cluster from the Nuclear centroid
ggplot(er_cluster_summary, aes(x = cluster_centroid_x, y = cluster_centroid_y, color = factor(cluster))) +
  geom_point(size = 3, alpha = 0.8) +
  facet_wrap(~ time) +
  scale_color_viridis_d() +
  labs(x = "Cluster X", y = "Cluster Y", color = "Cluster") +
  theme_minimal()

#nucleus Centroid Plot 
nucleus_centroids_plot <- nucleus_centroids %>%
  select(z, time, x_nuc, y_nuc) %>%
  mutate(label = paste0("Z", z, "_T", time))

# Make sure er_cluster is a factor
er_clustered <- er_clustered %>%
  mutate(er_cluster = as.factor(er_cluster))
#Filtering Cluster data by distance manually 
distance_threshold_px <- 25 / 6.142  #  4.07 pixels
distance_threshold_px <- ceiling(25 / 6.142)  # 5 pixels
er_clustered_filtered <- er_clustered %>%
  filter(dist_to_nucleus <= 150)


#filtering via histogram 
hist_data <- hist(er_clustered$dist_to_nucleus, breaks = 100, plot = FALSE)
counts <- hist_data$counts
gaps <- diff(counts)
gap_threshold <- -max(counts) * 0.1  # significant drop

gap_index <- which(gaps < gap_threshold)[1]
cutoff_distance <- hist_data$breaks[gap_index + 1]

er_clustered_filtered <- er_clustered %>%
  filter(dist_to_nucleus <= cutoff_distance)

plot(hist_data, main = "Distance to Nucleus Histogram")
abline(v = cutoff_distance, col = "red", lwd = 2, lty = 2)

ggplot(filter(er_clustered, time == 1), aes(x = x, y = y)) +
  geom_raster(aes(fill = er_cluster)) +
  geom_point(data = filter(nucleus_centroids_plot, time == 1), 
             aes(x = x_nuc, y = y_nuc),
             color = "blue", shape = 1, size = 2, stroke = 1) +
  coord_equal() +
  facet_wrap(~ z, ncol = 5) +
  scale_fill_viridis_d(na.value = "black") +
  scale_x_continuous(breaks = seq(0, 400, by = 100)) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 6),
    axis.text.y = element_text(size = 6),
    strip.text = element_text(face = "bold"),
    legend.position = "right"
  ) +
  labs(title = "ER Clusters by Z (Time 1)", fill = "ER Cluster")

ggplot(filter(er_clustered_filtered, time == 1), aes(x = x, y = y)) +
  geom_raster(aes(fill = er_cluster)) +
  geom_point(data = filter(nucleus_centroids_plot, time == 1), 
             aes(x = x_nuc, y = y_nuc),
             color = "blue", shape = 1, size = 2, stroke = 1) +
  coord_equal() +
  facet_wrap(~ z, ncol = 5) +
  scale_fill_viridis_d(na.value = "black") +
  scale_x_continuous(breaks = seq(0, 400, by = 100)) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 6),
    axis.text.y = element_text(size = 6),
    strip.text = element_text(face = "bold"),
    legend.position = "right"
  ) +
  labs(title = "ER Clusters by Z (Time 1)", fill = "ER Cluster")

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

plot(testr)

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

h2o.shutdown
