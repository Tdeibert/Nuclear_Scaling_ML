# Loading necessary libraries
library(dbscan)
library(data.table)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(readxl)
library(ggpmisc)
library(viridis)

# Load dataset
df <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/T10.csv")

df <- read_xlsx("D:/CDR2 for Analysis/CDR2/Area analysis.xlsx", sheet = "8.2 Macros")


#### Filtering Critera for Raw Data ####
# Extract X and Y columns for clustering
xy_data <- df[, c("XM", "YM")]

# Run DBSCAN clustering
db <- dbscan(xy_data, eps = 5, minPts = 1)
print(db)

# Assign clusters to original dataframe
df$cluster <- db$cluster

# Filter out noise (-1) if necessary
filtered_data <- subset(df, cluster >= 0)

# Initialize results storage
results <- list()

# Function to assign a region based on X, Y ranges
assign_region <- function(x, y) {
  if (x < 1900 & y < 1945) {
    return(1)
  } else if (x >= 1901 & x < 3801& y < 1945) {
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

#### Quantification of Filtered Data ####
# Loop over clusters
for (cluster_id in 1:53) {
  
  # Select nuclei in the current cluster
  nuclei <- subset(df, cluster == cluster_id)
  
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
    
    # Defining variables based on sorted nuclei
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
    
    Nuclear_Membrane_BG_Subtracted <- (((Membrane_int_H3 - Membrane_int_H1) - ((Nuclear_area_H3 - Nuclear_area_H1)) * (Membrane_int_H1 / Nuclear_area_H1)) / (2 * pi * sqrt(Nuclear_area_H2/pi)))*6.1538^2
    
    # Nuclear Pore Background Subtraction Calculation
    Nuclear_Pores <- ((NPC_int_H3 - NPC_int_H1) -
                        ((Nuclear_area_H3 - Nuclear_area_H1) * (NPC_int_H1 / Nuclear_area_H1)))
    
    Nuclear_Pore_BG_Subtracted <- (((NPC_int_H3 - NPC_int_H1) - ((Nuclear_area_H3 - Nuclear_area_H1)) * (NPC_int_H1 / Nuclear_area_H1)) / (2 * pi * sqrt(Nuclear_area_H2/pi)))*6.1538^2
    
    # Store results
    results[[as.character(cluster_id)]] <- data.frame(
      cluster = cluster_id,
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
    print(paste("Not enough nuclei detected in cluster", cluster_id))
  }
}

# Combine results into a single dataframe
final_results <- do.call(rbind, results)


#### Proccessing of the Final Results ####
final_results<- final_results %>% 
  mutate(SD = sd(Nuclear_Membrane_BG_Subtracted))

#reassigning my regions to the correct time interval. 
final_results <- final_results %>%
  mutate(Region = case_when(
    Region == 1 ~ 90,
    Region == 2 ~ 91,
    Region == 3 ~ 92,
    Region == 4 ~ 93,
    Region == 5 ~ 94,
    Region == 6 ~ 95,
    TRUE ~ Region  # Keeps other values unchanged
  ))

# Print results
print(final_results)


# Compute mean values per region
final_results_mean <- final_results %>%
  group_by(Region) %>%
  summarise(mean_Nuclear_Intensity = mean(Nuclear_Membrane_BG_Subtracted, na.rm = TRUE))


#generating a single time point into a variable
Time_point_14<- final_results


#### Cleaning Full Time Set#### 
#Saving Raw Unfiltered Data 
write.csv(Time_Point_1_10, "C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_3/Time_Point_1_10.csv")

#Merging Individual Time Points into a Single Data Set 
Time_Point_1_10 <- bind_rows(Time_point_1, Time_point_2, Time_point_3, Time_point_4, Time_point_5,Time_point_6,Time_point_7,Time_point_8,Time_point_9,Time_point_10, Time_point_11, Time_point_12, Time_point_13, Time_point_14)

#Generating a Working Data Set with Precise Columns
Full_Data_Set <- Time_Point_1_10 %>% 
  select(cluster, N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area, Slice, X, Y, Region) %>%  
  rename(Time = Region)

write.csv(Full_Data_Set, "D:/CDR2 for Analysis/CDR2/full_data_set.csv")


#### Interrogating the Data for Filtering ####
Full_Data_Set %>%
  summarise(
    min_value  = min(Nuclear_Membrane_BG_Subtracted[Nuclear_Membrane_BG_Subtracted > 0], na.rm = TRUE),
    max_value  = max(Nuclear_Membrane_BG_Subtracted, na.rm = TRUE),
    mean_value = mean(Nuclear_Membrane_BG_Subtracted, na.rm = TRUE)
  )


Full_Data_Set <- Full_Data_Set %>%
  filter(
    Nuclear_Membrane_BG_Subtracted >= 35,
    Nuclear_Membrane_BG_Subtracted <= 400000
  )

Full_Data_Set <- Full_Data_Set %>%
  filter(
    between(Nuclear_Membrane_BG_Subtracted, 35, 70000),
    !(Time > 60 & Nuclear_Membrane_BG_Subtracted < 20)
  )

#### Generating Plots of the Data #### 

#membrane Normalized Plot 
ggplot(data = Full_Data_Set, aes(x = Time, y = Nuclear_Membrane_BG_Subtracted, color = Nuclear_Membrane_BG_Subtracted)) +
  geom_point(size = 4) +
  scale_color_viridis_c(option = "magma") +  # Use scale_color_viridis_c() for points
  theme_minimal()+
  geom_smooth(method = "lm", color = "black", se = TRUE) +  # Linear regression
  labs(x = "Time in Minutes", y= "IntDen Membrane Normalized", title = "Nuclear Membrane Control")+
  stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
               formula = y ~ x, parse = TRUE, label.x.npc = "right", label.y.npc = 0.15)
#Nulcear Pore Plot 
#ggplot(data = Full_Data_Set, aes(x = Time, y = Nuclear_Pore_BG_Subtracted, color = Nuclear_Pore_BG_Subtracted)) +
 # geom_point(size = 4) +
  #scale_color_viridis_c(option = "magma") +  # Use scale_color_viridis_c() for points
  #theme_minimal()+
  #geom_smooth(method = "lm", color = "black", se = TRUE) +  # Linear regression
  #labs(x = "Time in Minutes", y= "IntDen NPC Normalized", title = "Nuclear Pore Control")+
  #stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
   #            formula = y ~ x, parse = TRUE, label.x.npc = "right", label.y.npc = 0.15)

#Nuclear Area Plot 
ggplot(data = Full_Data_Set, aes(x = Time, y = Nuclear_Area, color = Nuclear_Area)) +
  geom_point(size = 3) +
  scale_color_viridis_c(option = "magma") +  # Use scale_color_viridis_c() for points
  theme_minimal()+
  geom_smooth(method = "loess", span = 0.5, se = TRUE, color = "black") +
  labs(x = "Time in Minutes", y= "Nuclear Cross Sectional Area", title = "Nuclear Area Control")

#N/C Plot 
ggplot(data = Full_Data_Set, aes(x = Time, y = N_C_V2, color = N_C_V2)) +
  geom_point(size = 4) +
  scale_color_viridis_c(option = "magma") +
  theme_minimal() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3, raw = TRUE), color = "black", se = TRUE) +
  labs(x = "Time in Minutes", y = "N/C Ratio", title = "N/C Ratio")



#Better Statistical Models Plots 
exp_model <- nls(Full_Data_Set$N_C_V2 ~ A * (1 - exp(-k * Time)), 
                 data = Full_Data_Set, 
                 start = list(A = max(Full_Data_Set$N_C_V2), k = 0.1))

# Generate fitted values
Full_Data_Set$fit <- predict(exp_model)

# Plot Using the statitsical Model 
ggplot(Full_Data_Set, aes(x = Time, y = N_C_V2, color = N_C_V2)) +
  geom_point(size = 4) +
  geom_line(aes(y = fit), color = "black", size = 1.2) +  # Fitted curve
  scale_color_viridis_c(option = "magma") +
  theme_minimal() +
  labs(x = "Time in Minutes", y = "N/C Ratio", title = "N/C Ratio (Exponential Plateau Model)")



#### Duel Axis Plots #### 

#area + N/C 
max_nc <- max(Full_Data_Set$N_C_V2, na.rm = TRUE)
max_area <- max(Full_Data_Set$Nuclear_Area, na.rm = TRUE)
scale_factor <- max_nc / max_area

ggplot(Full_Data_Set, aes(x = Time)) +
  # Primary Y axis: N/C V2
  geom_point(aes(y = N_C_V2, color = N_C_V2), size = 2) +
  geom_line(aes(y = fit), color = "black", size = 1.2) +
  
  # Secondary Y axis: Area (scaled)
  geom_point(aes(y = Nuclear_Area * scale_factor), shape = 20, color = "black", size = 2, alpha = 0.9) +
  geom_smooth(aes(y = Nuclear_Area * scale_factor), method = "loess", se = FALSE,
              color = "black", linetype = "solid", size = 1.2) +
  
  # Define axes
  scale_y_continuous(
    name = "N/C Ratio",
    sec.axis = sec_axis(~ . / scale_factor, name = "Area")
  ) +
  scale_x_continuous(
    breaks = seq(0, max(Full_Data_Set$Time, na.rm = TRUE), by = 6),
    name   = "Time in Minutes"
  ) +
  scale_color_viridis_c(option = "magma", name = "N/C Ratio") +
  
  theme_minimal() +
  labs(title = "N/C Ratio and Area")




# Area + Membrane 
max_Membrane <- max(Full_Data_Set$Nuclear_Membrane_BG_Subtracted, na.rm = TRUE)
max_area <- max(Full_Data_Set$Nuclear_Area, na.rm = TRUE)
scale_factor <- max_Membrane / max_area

ggplot(Full_Data_Set, aes(x = Time)) +
  # Membrane intensity points (primary Y-axis)
  geom_point(aes(y = Nuclear_Membrane_BG_Subtracted), color = "darkgreen", size = 2) +
  
  # Membrane intensity fitted curve
  geom_smooth(
    aes(y = Nuclear_Membrane_BG_Subtracted),
    method = "lm",
    se = FALSE,
    color = "black",
    linetype = "solid",
    size = 1.2
  ) +
  
  # Fit line (if already calculated externally, like an exponential model)
  geom_line(aes(y = fit), color = "black", size = 1.2) +
  
  # Nuclear area scatter + fit on secondary Y-axis
  geom_point(aes(y = Nuclear_Area * scale_factor), shape = 18, color = "black", size = 2, alpha = 0.5) +
  geom_smooth(aes(y = Nuclear_Area * scale_factor), method = "loess", se = FALSE,
              color = "black", linetype = "solid", size = 1.2) +
  
  # Axes definitions
  scale_y_continuous(
    name = "Integrated Membrane Intensity",
    sec.axis = sec_axis(~ . / scale_factor, name = "Area")
  ) +
  scale_x_continuous(
    breaks = seq(0, max(Full_Data_Set$Time, na.rm = TRUE), by = 6),
    name   = "Time in Minutes"
  ) +
  
  theme_minimal() +
  labs(title = "Membrane Intensity and Nuclear Area")

####box and whisker plots ####

# Bin time into 6-min intervals
df_binned <- Full_Data_Set %>%
  mutate(
    Time_bin_num = floor(Time / 6) * 6,
    Time_bin_num = if_else(Time_bin_num < 6, 6, Time_bin_num),   # optional: start at 6
    Time_bin = factor(
      Time_bin_num,
      levels = seq(6, floor(max(Time, na.rm = TRUE) / 6) * 6, by = 6)
    )
  )

# Box and Whisker Plot of Binned Datasets
ggplot(df_binned, aes(x = Time_bin, y = Nuclear_Area)) +
  # Boxplots
  geom_boxplot(
    width = 0.6, alpha = 0.7, color = "black", fill = "grey",
    outlier.shape = NA
  ) +
  # Raw points
  geom_jitter(
    width = 0.2, color = "black", size = 1.5, alpha = 0.6
  ) +
  labs(
    title = "Nuclear Area CDR2",
    x = "Time Cluster (min)",
    y = "Nuclear Area"
  ) +
  theme_minimal(base_size = 13)
