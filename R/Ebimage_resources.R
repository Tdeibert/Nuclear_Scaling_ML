library(EBImage)
library(tidyverse)
# Load an image (any standard format, TIFF, PNG, etc.)
img <- readImage("C:/Users/tdeibert/.Working_Docs_Folder/UWYO/Nuclear Scaling Project/Data Sets/unmotified.tif")

#### Usefull interrogations about the image #### 
dim(img)        # expect something like (x, y, 20) if Z got read as channels
colorMode(img)  # should be 0 for grayscale

#### Functions to make images Read into R if meta data is incompatable #### 
# 1) Force grayscale (fix wrong color flag)
if (colorMode(img) == 2) colorMode(img) <- Grayscale  # or: colorMode(img) <- 0

# 2) If Z slices are sitting in the 3rd dim, convert them to frames (4th dim)
if (length(dim(img)) == 3 && dim(img)[3] > 1) {
  arr <- as.array(img)  # dims: x, y, z
  img <- Image(arr, dim = c(dim(arr)[1], dim(arr)[2], 1, dim(arr)[3]), colormode = Grayscale)
}

# 3) Now this will work:
display(normalize(img), all = TRUE)

display(img)             # View image
numberOfFrames(img)      # Number of frames (Z or T slices)
colorMode(img)           # 0 = grayscale, 2 = color
range(img)               # Min/max pixel intensity


img[100, 150, 1, 1]      # Pixel intensity at (x=100, y=150)
summary(as.vector(img))  # Quick stats of all pixel values


img_norm <- normalize(img)        # Scale to [0,1]
img_log  <- log(1 + img)          # Enhance faint signal
img_gamma <- img ^ 0.5            # Gamma correction


img_contrast <- equalize(img)     # Histogram equalization

blurred <- gblur(img, sigma = 2)               # Gaussian blur
medianed <- medianFilter(img, size = 3)        # Median smoothing
edges <- edge(img, method= "sobel")           # Edge detection built in didn't work wrote function
edges <- edgeCanny(img, sigma = 1)             # Canny edges didn't work 

bw <- img > 0.5                                # Simple threshold
bw_filled <- fillHull(bw)                      # Fill holes in objects
bw_opened <- opening(bw, makeBrush(5, "disc")) # Remove small specks
bw_closed <- closing(bw, makeBrush(5, "disc")) # Close small gaps
bw_dilated <- dilate(bw, makeBrush(3, "disc"))
bw_eroded  <- erode(bw, makeBrush(3, "disc"))

gray <- channel(img, "gray")

# Otsu threshold (adaptive to histogram)
thresh_val <- otsu(gray)
bw <- gray > thresh_val

# Adaptive/local threshold
bw_local <- thresh(gray, w = 15, h = 15, offset = 0.05)

# Label connected objects
labeled <- bwlabel(bw)

# Display results
display(colorLabels(labeled))

# Shape features
shape_feats <- computeFeatures.shape(labeled)

# Basic intensity features (area, mean, max, etc.)
intensity_feats <- computeFeatures.basic(labeled, img)

# Combine results
features <- data.frame(shape_feats, intensity_feats)
head(features)

masked <- img * bw                         # Mask image to binary region
object1 <- img * (labeled == 1)            # Isolate one object

# Extract a single Z slice or frame
slice5 <- getFrame(img, 5)

# Loop through all frames
for (i in seq_len(numberOfFrames(img))) {
  display(getFrame(img, i))
}

# Combine multiple images into a montage
montage(list(img, bw, colorLabels(labeled)), ncol = 3)


#### Function Building for Analysis#### 
# Edge Detection 
sobel_x <- matrix(c(-1,0,1,
                    -2,0,2,
                    -1,0,1), nrow = 3, ncol = 3)

sobel_y <- matrix(c(-1,-2,-1,
                    0, 0, 0,
                    1, 2, 1), nrow = 3, ncol = 3)

gx <- filter2(img, sobel_x)
gy <- filter2(img, sobel_y)

# Gradient magnitude
edges <- sqrt(gx^2 + gy^2)
display(normalize(edges))

#### Isolating a single slice from the image#### 
slice41 <- getFrame(img, 41)
display(slice41)
slice41 <- equalize(slice41)
display(slice41)
slice41 <- normalize(slice41) #effectively auto contrast
#manual Brightness and Contrast 
gain <- 1.0      # >1 increases contrast; <1 reduces
offset <- 0.0005   # >0 brightens; <0 darkens
slice41_bc <- slice41 * gain + offset
slice41_bc[slice41_bc < 0] <- 0
slice41_bc[slice41_bc > 1] <- 1
display(slice41_bc)

#### Background Subtraction #### 
bg <- gblur(slice41, sigma = 20)     # large sigma estimates background
slice41_bs <- slice41 - bg
slice41_bs[slice41_bs < 0] <- 0
slice41_enh <- normalize(slice41_bs)      # or do the 2–98% stretch again
display(slice41_enh)

#### Playing around with brightness and Contrast functions #### 
view_window <- function(x, lo = NULL, hi = NULL, p_lo = NULL, p_hi = NULL, clip = TRUE) {
  stopifnot(is.Image(x))
  g <- channel(x, "gray")  # display is usually grayscale
  a <- as.numeric(g)
  
  # Choose window by absolute values OR by percentiles (recommended)
  if (!is.null(p_lo) || !is.null(p_hi)) {
    if (is.null(p_lo)) p_lo <- 0
    if (is.null(p_hi)) p_hi <- 1
    qs <- quantile(a, c(p_lo, p_hi), na.rm = TRUE)
    lo <- qs[1]; hi <- qs[2]
  } else {
    if (is.null(lo) || is.null(hi)) stop("Provide lo/hi or p_lo/p_hi")
  }
  if (hi <= lo) stop("`hi` must be > `lo`")
  
  # Linear map [lo,hi] -> [0,1]
  y <- (g - lo) / (hi - lo)
  if (clip) { y[y < 0] <- 0; y[y > 1] <- 1 }
  
  y
}

#window selection 
display(view_window(slice41, p_lo = 0.01, p_hi = 0.99))

display(view_window(slice41, lo = 300, hi = 4000))

nf <- numberOfFrames(img)
for (i in seq_len(nf)) {
  sl <- getFrame(img, i)
  display(view_window(sl, p_lo = 0.02, p_hi = 0.98))
}

# Example (single slice sl):
# sl_view <- view_window(sl, p_lo = 0.02, p_hi = 0.98)  # robust window
# display(sl_view)
# ...but do measurements on `sl` (the original), not `sl_view`.