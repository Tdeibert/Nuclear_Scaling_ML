#Leavy Lab colaboration for image analysis. 
#required packages 
require(tidyverse)
require(dbscan)

#reading in your data for analysis. 
df <- read.csv("insert the pathway to your file here make sure to only include forward slashes in the file path")
  
#extracting your positional data from fiji data we will create a new object to hold the X&Y positions that dbscan will use to assign clustering values. 
x_y_positions <- df[c("XM", "YM")] %>% 

#use dbscan to generate clustering values. 
#how to use dbscan. eps value is the tolerance for how close X and Y can be and still be considered a single cluster value. The bigger the number the bigger the tollerence.   
#nuclei with large amounts of movement will need a larger value and nuclei that are close to eachother will need a smaller value. 
#ideally when you print DB you will have the same number of cluster values with the number total number of time points below it. 
db <- dbscan(x_y_positions, eps = 1, minPts = 1)

#we must now add the clustered values back into the original data set. 
df$cluster<- db$cluster 

# Convert Slice to Time Point (assuming 1 slice = 0.5 sec, modify as needed)
df$time_point <- df$Slice * 0.5

#at this point we now have our original data and a clustering value that will let us group together nuclei based on X and Y position using the clustering value. 
#we can now use this value to extract data from our original data frame. 

# Select required columns and rename
final_results <- df %>%
  select(cluster, time_point, XM, YM, Area, IntDen) %>%
  rename(X = XM, Y = YM) %>%
  arrange(cluster, time_point)  # Sort first by cluster, then by time

# Display the final results
print(final_results)
