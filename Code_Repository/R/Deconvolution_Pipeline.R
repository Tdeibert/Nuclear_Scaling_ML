library(terra)
library(tidyverse)
library(ggplot2)
library(tibble)
library(glue)


#### Load and prepare raster stack ####
raster_path <- "./Test_Stack.tif"
img <- rast(raster_path)

# Generate informative layer nam
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

#### Utility Functions ####


test_layer <- img[[layer_info$layer_name[45]]]
img_mat <- as.matrix(test_layer)
deconv_mat <- richardson_lucy_matrix(img_mat, psf, iterations = 15)

conv2_same <- function(image, kernel) {
  pad_rows <- nrow(image) + nrow(kernel) - 1
  pad_cols <- ncol(image) + ncol(kernel) - 1
  
  pad_image  <- matrix(0, pad_rows, pad_cols)
  pad_kernel <- matrix(0, pad_rows, pad_cols)
  pad_image[1:nrow(image), 1:ncol(image)] <- image
  pad_kernel[1:nrow(kernel), 1:ncol(kernel)] <- kernel
  
  # FFT convolution
  fft_image  <- fft(pad_image)
  fft_kernel <- fft(pad_kernel)
  conv_full  <- Re(fft(fft_image * fft_kernel, inverse = TRUE)) / (pad_rows * pad_cols)
  
  # Calculate crop coordinates
  center_row <- ceiling(nrow(kernel) / 2)
  center_col <- ceiling(ncol(kernel) / 2)
  
  row_start <- center_row
  col_start <- center_col
  row_end   <- row_start + nrow(image) - 1
  col_end   <- col_start + ncol(image) - 1
  
  # Crop and return exact size
  conv_same <- conv_full[row_start:row_end, col_start:col_end]
  return(conv_same)
}

generate_psf <- function(size = 15, sigma = 2) {
  x <- seq(-floor(size / 2), floor(size / 2), length.out = size)
  gauss <- outer(x, x, function(x, y) exp(-(x^2 + y^2) / (2 * sigma^2)))
  psf <- gauss / sum(gauss)
  return(psf)
}

# Richardson-Lucy Deconvolution on matrix
richardson_lucy_matrix <- function(image, psf, iterations = 10) {
  estimate <- image
  psf_mirror <- psf[nrow(psf):1, ncol(psf):1]
  
  for (i in 1:iterations) {
    convolved     <- conv2_same(estimate, psf)
    relative_blur <- image / (convolved + 1e-10)
    correction    <- conv2_same(relative_blur, psf_mirror)
    estimate      <- estimate * correction
  }
  
  return(estimate)
}
psf <- generate_psf(size = 21, sigma = 3)
deconv_mat <- richardson_lucy_matrix(img_mat, psf, iterations = 15)

#### Converting back into raster####
# Get reference layer
ref_layer <- img[[layer_info$layer_name[1]]]

# Check matrix size
dim(img_mat)        # original matrix from raster
dim(deconv_mat)     # after deconvolution

# Fix: manually assign matrix to a new raster with same dimensions
deconv_raster <- ref_layer  # clone spatial structure
values(deconv_raster) <- as.vector(deconv_mat)  # flatten matrix to vector

# Plot original and deconvolved
plot(c(ref_layer, deconv_raster), main = c("Original", "Deconvolved"))

####Single Layer Deconvolution####
deconvolve_raster_layer <- function(img, layer_name, psf, iterations = 10) {
  message(glue("Processing layer: {layer_name}"))
  
  r_layer <- img[[layer_name]]
  mat     <- as.matrix(r_layer)
  
  deconv_mat <- richardson_lucy_matrix(mat, psf, iterations)
  
  # Convert back to raster
  r_deconv <- rast(deconv_mat, crs = crs(r_layer))
  ext(r_deconv) <- ext(r_layer)
  
  names(r_deconv) <- paste0(layer_name, "_deconv")
  return(r_deconv)
}

#### Decon on all Layers #### 

# Example: only deconvolve Channel 1 across all Z and Time
target_layers <- layer_info %>%
  filter(channel == 1) %>%
  pull(layer_name)

# PSF
psf <- generate_psf(size = 21, sigma = 2)

# Run deconvolution
deconvolve_raster_layer <- function(img, layer_name, psf, iterations = 10) {
  message(glue("Processing layer: {layer_name}"))
  
  r_layer <- img[[layer_name]]
  mat     <- as.matrix(r_layer)
  deconv_mat <- richardson_lucy_matrix(mat, psf, iterations)
  
  # Force match structure with original layer
  r_deconv <- r_layer  # copy metadata
  values(r_deconv) <- as.vector(deconv_mat)  # apply deconvolved values
  
  names(r_deconv) <- paste0(layer_name, "_deconv")
  return(r_deconv)
}

# Redefine target layers
target_layers <- layer_info %>%
  filter(channel == 1) %>%
  pull(layer_name)

# PSF
psf <- generate_psf(size = 21, sigma = 3)

# Deconvolve again
deconvolved_layers <- lapply(target_layers, function(lyr) {
  deconvolve_raster_layer(img, lyr, psf, iterations = 15)
})
deconv_stack <- rast(deconvolved_layers)
####Plotting the Raster
dim(orig_layer)
dim(deconv_layer)

# Select a layer index to compare (e.g., first layer)
layer_index <- 1
orig_layer <- img[[layer_info$layer_name[layer_index]]]
deconv_layer <- deconv_stack[[layer_index]]

plot(c(orig_layer, deconv_layer), main = c("Original", "Deconvolved"))



#### 3D Deconvolution ####
# Get 3D volume for a single channel/time
get_3d_volume <- function(img, layer_info, timepoint = 1, channel = 1) {
  z_layers <- layer_info %>%
    filter(time == timepoint, channel == channel) %>%
    arrange(z) %>%
    pull(layer_name)
  
  vol_stack <- img[[z_layers]]
  vol_array <- as.array(vol_stack)  # [x, y, z]
  
  list(array = vol_array, layer_names = z_layers, stack = vol_stack)
}

# 3D Gaussian PSF
generate_3d_psf <- function(size = c(15, 15, 7), sigma = c(2, 2, 1)) {
  x <- seq(-floor(size[1]/2), floor(size[1]/2), length.out = size[1])
  y <- seq(-floor(size[2]/2), floor(size[2]/2), length.out = size[2])
  z <- seq(-floor(size[3]/2), floor(size[3]/2), length.out = size[3])
  
  psf3d <- array(0, dim = size)
  for (i in seq_along(x)) {
    for (j in seq_along(y)) {
      for (k in seq_along(z)) {
        psf3d[i, j, k] <- exp(
          -(x[i]^2 / (2*sigma[1]^2) +
              y[j]^2 / (2*sigma[2]^2) +
              z[k]^2 / (2*sigma[3]^2))
        )
      }
    }
  }
  psf3d / sum(psf3d)
}

# 3D FFT convolution
fft_convolve3d <- function(volume, psf) {
  vol_fft <- fft(volume)
  psf_fft <- fft(psf, dim(volume))
  Re(fft(vol_fft * psf_fft, inverse = TRUE) / length(vol_fft))
}

# Richardson-Lucy in 3D
richardson_lucy_3d <- function(image, psf, iterations = 10) {
  estimate <- image
  psf_mirror <- psf[rev(seq_len(dim(psf)[1])),
                    rev(seq_len(dim(psf)[2])),
                    rev(seq_len(dim(psf)[3]))]
  
  for (i in 1:iterations) {
    blurred <- fft_convolve3d(estimate, psf)
    relative_blur <- image / (blurred + 1e-10)
    correction <- fft_convolve3d(relative_blur, psf_mirror)
    estimate <- estimate * correction
  }
  estimate
}

# 3D convolution using FFT
fft_convolve3d <- function(volume, psf) {
  vol_fft <- fft(volume)
  psf_fft <- fft(psf, dim(volume))
  Re(fft(vol_fft * psf_fft, inverse = TRUE) / length(vol_fft))
}

richardson_lucy_3d <- function(image, psf, iterations = 10) {
  estimate <- image
  psf_mirror <- psf[rev(seq_len(dim(psf)[1])),
                    rev(seq_len(dim(psf)[2])),
                    rev(seq_len(dim(psf)[3]))]
  
  for (i in 1:iterations) {
    blurred     <- fft_convolve3d(estimate, psf)
    relative_blur <- image / (blurred + 1e-10)
    correction  <- fft_convolve3d(relative_blur, psf_mirror)
    estimate <- estimate * correction
  }
  
  return(estimate)
}

volume_to_stack <- function(volume_array, ref_stack, layer_names) {
  slices <- lapply(seq_len(dim(volume_array)[3]), function(i) {
    r <- rast(volume_array[, , i], crs = crs(ref_stack[[i]]))
    ext(r) <- ext(ref_stack[[i]])
    names(r) <- paste0(layer_names[i], "_deconv")
    return(r)
  })
  
  rast(slices)
}


# 1. Extract 3D volume for C1, T1
vol <- get_3d_volume(img, layer_info, timepoint = 1, channel = 1)

# 2. Generate 3D PSF (XY sigma = 2, Z sigma = 1)
psf3d <- generate_3d_psf(size = c(15, 15, 7), sigma = c(2, 2, 1))

# 3. Run 3D deconvolution
deconv_3d <- richardson_lucy_3d(vol$array, psf3d, iterations = 15)

# 4. Convert to terra raster
deconv_stack <- volume_to_stack(deconv_3d, vol$stack, vol$layer_names)

# 5. Save or plot
plot(deconv_stack[[1]])
writeRaster(deconv_stack, "Deconvolved_C1_T1_3D.tif", overwrite = TRUE)