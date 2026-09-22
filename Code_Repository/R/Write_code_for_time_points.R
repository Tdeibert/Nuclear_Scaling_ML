# Loading necessary libraries
library(dbscan)
library(data.table)
library(dplyr)
library(ggplot2)
library(tidyverse)

# Load dataset
df <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/T4.csv")

# Extract X and Y columns for clustering
xy_data <- df[, c("XM", "YM")]

# Run DBSCAN clustering
db <- dbscan(xy_data, eps = 30, minPts = 1)
print(db)

# Assign clusters to original dataframe
df$cluster <- db$cluster

# Filter out noise (-1) if necessary
df <- df[df$cluster >= 0, ]

# Initialize results storage
results <- list()

# Function to assign a region based on X, Y ranges
assign_region <- function(x, y) {
  if (x < 1900 & y < 1945) {
    return(1)
  } else if (x >= 1901 & x < 3801 & y < 1945) {
    return(2)
  } else if (x >= 3802 & x < 5732 & y < 1945) {
    return(3)
  } else if (x < 1900 & y >= 1945) {
    return(4)
  } else if (x >= 1901 & x < 3801 & y >= 1945) {
    return(5)
  } else {
    return(6)
  }
}

# Loop over unique clusters
for (cluster in unique(df$cluster)) {  
  # Select nuclei in the current cluster
  nuclei <- df[df$cluster == cluster, ]
  
  # Ensure the cluster is not empty
  if (nrow(nuclei) == 0) next
  
  # Identify the row with the maximum Area value in the current cluster
  max_area_row <- nuclei[which.max(nuclei$Area), ]
  
  # Extract the Slice value for the maximum Area
  slice_value <- max_area_row$Slice
  
  # Filter the current cluster dataframe to keep only rows with the identified Slice value
  nuclei <- nuclei[nuclei$Slice == slice_value, ]
  
  # Sorting by Area
  nuclei <- nuclei %>% arrange(Area)
  
  # Ensure there are at least 8 rows before proceeding
  if (nrow(nuclei) >= 8) {
    
    # Define variables based on sorted nuclei
    NLS_int_H1 <- nuclei$IntDen[1]
    NLS_int_H2 <- nuclei$IntDen[4]
    NLS_int_H3 <- nuclei$IntDen[5]
    NLS_int_H4 <- nuclei$IntDen[8]
    
    Nuclear_area_H1 <- nuclei$Area[1]
    Nuclear_area_H2 <- nuclei$Area[4]
    Nuclear_area_H3 <- nuclei$Area[5]
    Nuclear_area_H4 <- nuclei$Area[8]
    
    Membrane_int_H1 <- nuclei$m_IntDen[1]
    Membrane_int_H2 <- nuclei$m_IntDen[4]
    Membrane_int_H3 <- nuclei$m_IntDen[5]
    Membrane_int_H4 <- nuclei$m_IntDen[8]
    
    NPC_int_H1 <- nuclei$npc_IntDen[1]
    NPC_int_H2 <- nuclei$npc_IntDen[4]
    NPC_int_H3 <- nuclei$npc_IntDen[5]
    NPC_int_H4 <- nuclei$npc_IntDen[8]
    
    Membrane_mean <- nuclei$m_Mean[1]
    NPC_mean <- nuclei$npc_Mean[1]
    NLS_mean <- nuclei$Mean[4]
    X_cordinate <- nuclei$XM[1]
    Y_cordinate <- nuclei$YM[1]
    Nuclear_area <- (Nuclear_area_H2/6.1538^2) 
    
    # Assign region based on X, Y
    region <- assign_region(X_cordinate, Y_cordinate)
    
    # N/C Ratio Calculation
    N_C_V2 <- NLS_mean / (NLS_mean + (NLS_int_H4 - NLS_int_H3) / (Nuclear_area_H4 - Nuclear_area_H3))
    
    # Nuclear Membrane Background Subtraction Calculation
    Nuclear_Membrane <-((Membrane_int_H3 - Membrane_int_H1) - 
                          ((Nuclear_area_H3 - Nuclear_area_H1) * (Membrane_int_H1 / Nuclear_area_H1)))
    
    Nuclear_Membrane_BG_Subtracted <- ((Membrane_int_H3 - Membrane_int_H1) - 
                                         ((Nuclear_area_H3 - Nuclear_area_H1) * (Membrane_int_H1 / Nuclear_area_H1))) / 
      (2 * pi * sqrt((Nuclear_area_H2/6.1538^2) / pi))
    
    # Nuclear Pore Background Subtraction Calculation
    Nuclear_Pores <- ((NPC_int_H3 - NPC_int_H1) -
                        ((Nuclear_area_H3 - Nuclear_area_H1) * (NPC_int_H1 / Nuclear_area_H1)))
    
    Nuclear_Pore_BG_Subtracted <- ((NPC_int_H3 - NPC_int_H1) -
                                     ((Nuclear_area_H3 - Nuclear_area_H1) * (NPC_int_H1 / Nuclear_area_H1))) / 
      (2 * pi * sqrt((Nuclear_area_H2/6.1538^2) / pi))
    
    # Store results
    results[[as.character(cluster)]] <- data.frame(
      cluster = cluster,
      N_C_V2 = N_C_V2,
      Nuclear_Membrane = Nuclear_Membrane,
      Nuclear_Membrane_BG_Subtracted = Nuclear_Membrane_BG_Subtracted,
      Nuclear_Pores = Nuclear_Pores,
      Nuclear_Pore_BG_Subtracted = Nuclear_Pore_BG_Subtracted,
      Nuclear_Area = Nuclear_area,
      Slice = slice_value,
      X = X_cordinate,
      Y = Y_cordinate,
      Region = region
    )
    
  } else {
    print(paste("Not enough nuclei detected in cluster", cluster))
  }
}

# Check if results is empty before combining
if (length(results) == 0) {
  final_results <- data.frame()  # Create an empty data frame
} else {
  final_results <- do.call(rbind, results) %>% as.data.frame()
}

# Display the final results
print(final_results)



final_results<- final_results %>% 
  mutate(SD = sd(Nuclear_Membrane_BG_Subtracted))

#reassigning my regions to the correct time interval. 
final_results <- final_results %>%
  mutate(Region = case_when(
    Region == 1 ~ 16,
    Region == 2 ~ 17,
    Region == 3 ~ 18,
    Region == 4 ~ 19,
    Region == 5 ~ 20,
    Region == 6 ~ 22,
    TRUE ~ Region  # Keeps other values unchanged
  ))

Time_point_4 <- final_results

Time_point_4_clean <- Time_point_4 %>% 
  filter(if_all(everything(), ~. >=0))

write.csv(Time_point_4_clean,"C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs/Time_point_4.csv")
