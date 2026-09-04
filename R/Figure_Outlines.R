#combining Dataframes for plotting 
library(tidyverse)
library(ggplot2)
library(ggpmisc)
library(viridis)


Time_point_1 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_1.csv")
Time_point_2 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_2.csv")
Time_point_3 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_3.csv")
Time_point_4 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_4_2.csv")
Time_point_5 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_5.csv")
Time_point_6 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_6.csv")
Time_point_7 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_7.csv")
Time_point_8 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_8.csv")
Time_point_9 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_9.csv")
Time_point_10 <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_2/Time_point_10.csv")

Time_Point_1_10 <- bind_rows(Time_point_1, Time_point_2, Time_point_3, Time_point_4, Time_point_5,Time_point_6,Time_point_7,Time_point_8,Time_point_9,Time_point_10)


ggplot(data = Full_Data_Set, aes(x = Time, y = Nuclear_Membrane_BG_Subtracted, color = Nuclear_Membrane_BG_Subtracted)) +
  geom_point(size = 4) +
  scale_color_viridis_c(option = "magma") +  # Use scale_color_viridis_c() for points
  theme_minimal()+
  geom_smooth(method = "lm", color = "black", se = TRUE) +  # Linear regression
  labs(x = "Time in Minutes", y= "IntDen Membrane Normalized", title = "Nuclear Membrane Control")+
  stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
               formula = y ~ x, parse = TRUE, label.x.npc = "right", label.y.npc = 0.15)

ggplot(data = Full_Data_Set, aes(x = Time, y = Nuclear_Pore_BG_Subtracted, color = Nuclear_Pore_BG_Subtracted)) +
  geom_point(size = 4) +
  scale_color_viridis_c(option = "magma") +  # Use scale_color_viridis_c() for points
  theme_minimal()+
  geom_smooth(method = "lm", color = "black", se = TRUE) +  # Linear regression
  labs(x = "Time in Minutes", y= "IntDen NPC Normalized", title = "Nuclear Pore Control")+
  stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
               formula = y ~ x, parse = TRUE, label.x.npc = "right", label.y.npc = 0.15)

ggplot(data = Full_Data_Set, aes(x = Time, y = Nuclear_Area, color = Nuclear_Area)) +
  geom_point(size = 3) +
  scale_color_viridis_c(option = "magma") +  # Use scale_color_viridis_c() for points
  theme_minimal()+
  geom_smooth(method = "loess", span = 0.5, se = TRUE, color = "black") +
  labs(x = "Time in Minutes", y= "Nuclear Cross Sectional Area", title = "Nuclear Area Control")

ggplot(data = Full_Data_Set, aes(x = Time, y = N_C_V2, color = N_C_V2)) +
    geom_point(size = 4) +
    scale_color_viridis_c(option = "magma") +
    theme_minimal() +
    geom_smooth(method = "lm", formula = y ~ poly(x, 3, raw = TRUE), color = "black", se = TRUE) +
    labs(x = "Time in Minutes", y = "N/C Ratio", title = "N/C Ratio")


exp_model <- nls(Full_Data_Set$N_C_V2 ~ A * (1 - exp(-k * Time)), 
                 data = Full_Data_Set, 
                 start = list(A = max(Full_Data_Set$N_C_V2), k = 0.1))

# Generate fitted values
Full_Data_Set$fit <- predict(exp_model)

# Plot
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
  # Primary Y axis: N_C_V2 as points and fitted curve
  geom_point(aes(y = N_C_V2, color = N_C_V2), size = 2) +
  geom_line(aes(y = fit), color = "black", size = 1.2) +
  
  # Secondary Y axis: Area (scaled) as scatter points
  geom_point(aes(y = Nuclear_Area * scale_factor), shape = 20, color = "black", size = 2, alpha = 0.9) +
  
  # Fitted curve to Area (scaled)
  geom_smooth(aes(y = Nuclear_Area * scale_factor), method = "loess", se = FALSE, color = "black", linetype = "solid", size = 1.2) +
  
  # Define axes
  scale_y_continuous(
    name = "N/C Ratio",
    sec.axis = sec_axis(~ . / scale_factor, name = "Area")
  ) +
  scale_color_viridis_c(option = "magma", name = "N/C Ratio") +
  
  theme_minimal() +
  labs(
    x = "Time in Minutes",
    title = "N/C Ratio and Area"
  )
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
  geom_point(aes(y = Nuclear_Area * scale_factor), shape = 20, color = "black", size = 2, alpha = 0.9) +
  geom_smooth(aes(y = Nuclear_Area * scale_factor), method = "loess", se = FALSE, color = "black", linetype = "solid", size = 1.2) +
  
  # Axes definitions
  scale_y_continuous(
    name = "Integrated Membrane Intensity",
    sec.axis = sec_axis(~ . / scale_factor, name = "Area")
  ) +
  
  theme_minimal() +
  labs(
    x = "Time in Minutes",
    title = "Membrane Intensity and Nuclear Area"
  )


#plotly leaflet 
