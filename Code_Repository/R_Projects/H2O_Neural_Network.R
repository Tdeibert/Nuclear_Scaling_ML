#ChatGTP workflow for H2O unsupervised analysis. 
library(h2o)
library(terra)
library(data.table)

#intialize H2O
h2o.init(nthreads = -1, max_mem_size = "8G")  # Adjust memory as needed

#load Image
raster_image <- rast("./Test_Z_NPC.tif")  # Load the raster file

#convert to a dataframe
image_df <- as.data.frame(raster_image, xy = TRUE, na.rm = TRUE)  # Include X, Y coordinates
duplicated(colnames(image_df))
colnames(image_df) <- make.names(colnames(image_df), unique = TRUE)
any(duplicated(colnames(image_df))) 


#h20 image format 
image_h2o <- as.h2o(image_df)

#starting my undervervised analysis 
pca_model <- h2o.prcomp(image_h2o, k = 5, transform = "STANDARDIZE", impute_missing = TRUE)
summary(pca_model)

#cluster Pixels based on spectra 
kmeans_model <- h2o.kmeans(training_frame = image_h2o, k = 5, standardize = TRUE)
clusters <- h2o.predict(kmeans_model, image_h2o)
image_df$cluster <- as.vector(clusters)

#Feature extraction 
autoencoder_model <- h2o.deeplearning(
  x = names(image_df)[3:ncol(image_df)],  # Ignore X and Y
  training_frame = image_h2o,
  autoencoder = TRUE,
  hidden = c(50, 10, 50),  # Adjust based on complexity
  activation = "Tanh",
  epochs = 100
)

# Exclude 'cluster' column from features
features <- setdiff(names(image_df)[3:ncol(image_df)], "cluster")

autoencoder_model <- h2o.deeplearning(
  x = features,  # Use the corrected feature set
  training_frame = image_h2o,
  autoencoder = TRUE,
  hidden = c(50, 10, 50),  # Adjust based on complexity
  activation = "Tanh",
  epochs = 100
)

#extract features
features <- h2o.deepfeatures(autoencoder_model, image_h2o, layer = 2)

#turn it back into a raster. 
raster_image$cluster <- image_df$cluster
plot(raster_image, 1:15)
plot(raster_image, 16:20)
h2o.shutdown(prompt = FALSE)


