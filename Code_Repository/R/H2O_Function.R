# Load libraries
library(terra)
library(h2o)
library(tidyverse)
library(sf)

# Start H2O
h2o.init(max_mem_size = "8G")
raster_path <- "./Test_stack.tif"
# --- Otsu Threshold: Calculating the background intensity of the image ---
#nbins refers to the bitdepth of the histogram. 256 for 8 bit and 65536 for 16bit.
#calculate_otsu_threashold: Function definition: Takes a vector of pixel intensity values (intensity_values) and divides them into a specified number of histogram bins
#h <- hist(intensity_values, breaks = nbins, plot = FALSE): Function Definition: creates a histogram. 
#p <- h$counts / sum(h$counts): Function Definition: Converts the histogram into normalized probablilties for each bin. 
#omega <- cumsum(p): Function Definition: omega[k] is the cumulative probability of class 1 (foreground) up to bin k.
#mu <- cumsum(p * h$mids): Function Definition: mu[k] is the cumulative mean up to bin k.: average mean. 
#mu_t <- mu[length(mu)]: Function Definition: Total mean intensity of the image. 
#sigma2 <- (mu_t * omega - mu)^2 / (omega * (1 - omega)): Function Definition: computes the between class variance for each threashold. 

calculate_otsu_threshold <- function(intensity_values, nbins = 65536) {
  h      <- hist(intensity_values, breaks = nbins, plot = FALSE)
  p      <- h$counts / sum(h$counts)
  omega  <- cumsum(p)
  mu     <- cumsum(p * h$mids)
  mu_t   <- mu[length(mu)]
  sigma2 <- (mu_t*omega - mu)^2 / (omega * (1 - omega))
  h$mids[which.max(sigma2)]
}

# --- Master: process multi-channel/time Z-stack ---
process_microscopy_raster <- function(
    raster_path = "./Test_stack.tif",
    n_channels = 3,
    n_z = 20,
    n_time = 2,
    cluster_k             = 10,
    deep_learning_epochs  = 50
) {
  # Read the full stack
  img     <- rast(raster_path)
  n_layers <- nlyr(img)
  
  # Build a lookup dataframe: which layer → channel, z, time
  layer_info <- expand.grid(
    channel = 1:n_channels,
    z       = 1:n_z,
    time    = 1:n_time
  ) %>%
    arrange(time, z, channel) %>%
    mutate(layer = row_number())
  
  results <- list()
  
  for (i in seq_len(n_layers)) {
    li <- layer_info %>% filter(layer == i)
    ch <- li$channel
    zs <- li$z
    tm <- li$time
    
    message(glue::glue("Layer {i}: channel={ch}, z={zs}, time={tm}"))
    
    # 1) extract that single layer
    lyr <- img[[i]]
    vals <- as.numeric(values(lyr))
    coords <- xyFromCell(lyr, seq_along(vals))
    
    r_df <- tibble(
      x         = coords[,1],
      y         = coords[,2],
      intensity = vals
    ) %>%
      filter(!is.na(intensity)) %>%
      mutate(
        intensity = (intensity - min(intensity)) /
          (max(intensity) - min(intensity)) #helps normalize intensities for background removal. 
      )
    
    # 2) background removal
    thr  <- calculate_otsu_threshold(r_df$intensity)
    r_df <- filter(r_df, intensity > thr)
    if (nrow(r_df) < 10) {
      warning("… too few foreground pixels at layer ", i)
      next
    }
    
    # 3) autoencoder → deep features
    h2o_df <- as.h2o(r_df)
    ae <- h2o.deeplearning(
      x             = c("x","y","intensity"),
      training_frame= h2o_df,
      autoencoder   = TRUE,
      reproducible  = TRUE,
      seed          = 1234,
      epochs        = deep_learning_epochs,
      hidden        = c(50,20,50),
      activation    = "Tanh"
    )
    features <- h2o.deepfeatures(ae, h2o_df, layer = 2)
    
    # 4) cluster those features
    km       <- h2o.kmeans(training_frame = features, k = cluster_k, seed = 1234)
    clusters <- as.vector(h2o.predict(km, features)$predict)
    r_df$cluster <- clusters
    
    # 5) summarize per cluster
    summary_df <- r_df %>%
      group_by(cluster) %>%
      summarise(
        area                 = n(),
        integrated_intensity = sum(intensity),
        avg_intensity        = mean(intensity),
        channel              = ch,
        z_layer              = zs,
        time                 = tm,
        .groups = "drop"
      )
    
    results[[i]] <- summary_df
  }
  
  # 6) combine everything
  bind_rows(results)
}


results_single <- process_microscopy_raster()


write_csv(results_single, "./Test_Results.csv")


