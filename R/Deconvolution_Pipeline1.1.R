# Load required libraries
library(terra)
library(tidyverse)
library(ggplot2)
library(tibble)
library(glue)

#### USER-DEFINED PARAMETERS ####
# Microscope acquisition parameters
objective_na <- 1.4                # Example: 1.4 NA oil immersion
wavelength   <- 0.510              # Emission wavelength in microns (e.g. 510 nm)
pixel_size_xy <- 6.135           # XY pixel size in microns (camera pixel size × magnification)
z_spacing    <- 2               # Distance between Z-slices in microns
# In addition to the raster file and structural layout (channels, Z, time),
# microscope acquisition parameters are critical for accurate PSF modeling.
# The following microscopy parameters should be adjusted to match your system:
# - objective_na: Numerical aperture of the objective (affects PSF width)
# - wavelength: Wavelength of emitted light in microns (affects PSF width)
# - z_spacing: Distance between Z-slices in microns (affects 3D PSF aspect ratio)
#
# These are used to derive the sigma values for the PSF generation.

# Modify these based on your specific image acquisition setup:
raster_path <- "./Test_Stack.tif"   # Path to your multi-layer .tif image
n_channels  <- 3                     # Number of channels in your dataset
n_z         <- 20                    # Number of Z slices per timepoint
n_time      <- 2                     # Number of timepoints

#### Load and prepare raster stack ####
img <- rast(raster_path)
expected_layers <- n_channels * n_z * n_time
actual_layers   <- nlyr(img)

if (actual_layers != expected_layers) {
  stop(glue("Mismatch in layers: expected {expected_layers}, got {actual_layers}"))
}

# Generate informative layer names
layer_info <- expand_grid(
  time    = 1:n_time,
  z       = 1:n_z,
  channel = 1:n_channels
) %>%
  mutate(layer_index = row_number(),
         layer_name  = str_glue("T{time}_Z{z}_C{channel}"))

names(img) <- layer_info$layer_name

#### Utility Functions ####

# 2D convolution using FFT with same-size output
conv2_same <- function(image, kernel) {
  pad_rows <- nrow(image) + nrow(kernel) - 1
  pad_cols <- ncol(image) + ncol(kernel) - 1
  
  pad_image  <- matrix(0, pad_rows, pad_cols)
  pad_kernel <- matrix(0, pad_rows, pad_cols)
  pad_image[1:nrow(image), 1:ncol(image)] <- image
  pad_kernel[1:nrow(kernel), 1:ncol(kernel)] <- kernel
  
  fft_image  <- fft(pad_image)
  fft_kernel <- fft(pad_kernel)
  conv_full  <- Re(fft(fft_image * fft_kernel, inverse = TRUE)) / (pad_rows * pad_cols)
  
  center_row <- ceiling(nrow(kernel) / 2)
  center_col <- ceiling(ncol(kernel) / 2)
  row_start <- center_row
  col_start <- center_col
  row_end   <- row_start + nrow(image) - 1
  col_end   <- col_start + ncol(image) - 1
  
  conv_same <- conv_full[row_start:row_end, col_start:col_end]
  return(conv_same)
}

# 2D Gaussian PSF
generate_psf <- function(size = 15, sigma = 2) {
  x <- seq(-floor(size / 2), floor(size / 2), length.out = size)
  gauss <- outer(x, x, function(x, y) exp(-(x^2 + y^2) / (2 * sigma^2)))
  psf <- gauss / sum(gauss)
  return(psf)
}

# Richardson-Lucy deconvolution for 2D images
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

# Apply 2D deconvolution to a single raster layer with normalization
deconvolve_raster_layer <- function(img, layer_name, psf, iterations = 10) {
  message(glue("Processing layer: {layer_name}"))
  
  r_layer <- img[[layer_name]]
  mat     <- as.matrix(r_layer)
  deconv_mat <- richardson_lucy_matrix(mat, psf, iterations)
  
  # Normalize deconvolved matrix to 16-bit range (0–65535)
  deconv_mat <- (deconv_mat - min(deconv_mat)) / 
    (max(deconv_mat) - min(deconv_mat) + 1e-10) * 65535
  deconv_mat <- pmin(pmax(deconv_mat, 0), 65535)  # Clip to avoid overflows
  
  r_deconv <- r_layer  # Copy structure
  values(r_deconv) <- as.vector(deconv_mat)
  names(r_deconv) <- paste0(layer_name, "_deconv")
  return(r_deconv)
}

#### Batch 2D Deconvolution ####

target_layers <- layer_info %>%
  filter(channel == 1) %>%
  pull(layer_name)


# Derived PSF sigmas (microns per pixel assumed = 1 for simplicity; adjust if needed)
sigma_xy <- 0.21 * wavelength / objective_na            # lateral resolution
sigma_z  <- 0.66 * wavelength / (objective_na^2)        # axial resolution

theoretical_sigma <- sigma_xy  # for 2D deconvolution
psf <- generate_psf(size = 21, sigma = theoretical_sigma)

deconvolved_layers <- lapply(target_layers, function(lyr) {
  deconvolve_raster_layer(img, lyr, psf, iterations = 15)
})

deconv_stack <- rast(deconvolved_layers)

# Compare one original vs deconvolved layer
layer_index <- 16
orig_layer <- img[[layer_info$layer_name[layer_index]]]
deconv_layer <- deconv_stack[[layer_index]]
plot(c(orig_layer, deconv_layer), main = c("Original", "Deconvolved"))

writeRaster(
  deconv_stack,
  "C:/Users/tdeibert/OneDrive - University of Wyoming/Documents/UWYO/Nuclear Scaling Project/Data Sets/deconv_stack.tif",
  filetype = "GTiff",
  overwrite = TRUE,
  gdal = c("BIGTIFF=YES")
)

writeRaster(
  img,
  "C:/Users/tdeibert/OneDrive - University of Wyoming/Documents/UWYO/Nuclear Scaling Project/Data Sets/unmotified.tif",
  filetype = "GTiff",
  overwrite = TRUE,
  gdal = c("BIGTIFF=YES")
)
#### 3D Deconvolution Functions ####

get_3d_volume <- function(img, layer_info, timepoint = 1, channel = 1) {
  z_layers <- layer_info %>%
    filter(time == timepoint, channel == channel) %>%
    arrange(z) %>%
    pull(layer_name)
  
  vol_stack <- img[[z_layers]]
  vol_array <- as.array(vol_stack)
  
  list(array = vol_array, layer_names = z_layers, stack = vol_stack)
}

generate_3d_psf <- function(size = c(15, 15, 7),
                            objective_na = 1.4,
                            wavelength = 0.510,          # microns
                            pixel_size_xy = 0.108,        # microns
                            z_spacing = 0.3) {            # microns
  
  # Calculate sigma in microns
  sigma_xy_microns <- 0.21 * wavelength / objective_na
  sigma_z_microns  <- 0.66 * wavelength / (objective_na^2)
  
  # Convert sigma to pixels
  sigma_xy_pixels <- sigma_xy_microns / pixel_size_xy
  sigma_z_pixels  <- sigma_z_microns  / z_spacing
  
  # Coordinate grid in pixels
  x <- seq(-floor(size[1]/2), floor(size[1]/2), length.out = size[1])
  y <- seq(-floor(size[2]/2), floor(size[2]/2), length.out = size[2])
  z <- seq(-floor(size[3]/2), floor(size[3]/2), length.out = size[3])
  
  psf3d <- array(0, dim = size)
  for (i in seq_along(x)) {
    for (j in seq_along(y)) {
      for (k in seq_along(z)) {
        psf3d[i, j, k] <- exp(
          -(x[i]^2 / (2 * sigma_xy_pixels^2) +
              y[j]^2 / (2 * sigma_xy_pixels^2) +
              z[k]^2 / (2 * sigma_z_pixels^2))
        )
      }
    }
  }
  
  return(psf3d / sum(psf3d))
}


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
    ref_layer <- ref_stack[[i]]
    slice <- volume_array[, , i]
    
    r <- ref_layer  # clone structure
    values(r) <- as.vector(slice)
    names(r) <- paste0(layer_names[i], "_deconv")
    return(r)
  })
  
  rast(slices)
}

#### 3D Deconvolution Execution ####

# Extract 3D volume for C1, T1
vol <- get_3d_volume(img, layer_info, timepoint = 1, channel = 1)

# Generate 3D PSF based on microscope properties
psf3d <- generate_3d_psf(size = c(15, 15, 7),
                         objective_na = objective_na,
                         wavelength = wavelength,
                         pixel_size_xy = pixel_size_xy,
                         z_spacing = z_spacing)

# Run 3D deconvolution
deconv_3d <- richardson_lucy_3d(vol$array, psf3d, iterations = 15)

# Convert to raster stack
deconv_stack_3d <- volume_to_stack(deconv_3d, vol$stack, vol$layer_names)

# Plot or save
plot(deconv_stack_3d[[1]], main = "Deconvolved Z1")
writeRaster(deconv_stack_3d, "Deconvolved_C1_T1_3D.tif", overwrite = TRUE)

