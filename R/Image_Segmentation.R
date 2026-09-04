#New Code for establishing an Image Analysis Pipeline using R. 
require(terra)
require(tidyverse)
require(ggplot2)
require(h2o)
test_raster <- rast("./Test_Z_NPC.tif")
plot(test_raster)
plot(test_raster, 16:20)


#threasholding step 
threshold_value <- 300
#create a mask using my threashold value. 
binary_mask <- test_raster > threshold_value
plot(binary_mask)

maskdatafram <- as.data.frame(binary_mask)
  
  

#creating an integrated density for my data
sum_thresholded <- global(test_raster * (test_raster > threshold_value), sum, na.rm=TRUE)
print(sum_thresholded)
Threshold_Quantificaiton <- as.data.frame(sum_thresholded)
#that seems about correct for based on what I would measure with fiji. 

#using the mask to generate intesity values. 

masked_int <- test_raster * binary_mask
print(masked_int)

integrated_intensity <- global(masked_int, sum, na.rm=TRUE)
print(integrated_intensity)

#labeling the circles as seperate objects. 
objects <- patches(binary_mask, directions=8)  # Label each connected object
n_objects <- global(objects, max, na.rm=TRUE)  # Count the number of objects

# Extract intensity for each object
objectwise_intensity <- zonal(masked_int, objects, fun="sum", na.rm=TRUE)
print(objectwise_intensity)
