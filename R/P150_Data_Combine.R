library(tidyr)
library(tidyverse)
#Cleaning up processed P150 Data 
Time_Point_1 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T1_Final.csv")
Time_Point_2 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T2_Final.csv")
Time_Point_3 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T3_Final.csv")
Time_Point_4 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T4_Final.csv")
Time_Point_5 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T5_Final.csv")
Time_Point_6 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T6_Final.csv")
Time_Point_7 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T7_Final.csv")
Time_Point_8 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T8_Final.csv")
Time_Point_9 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T9_Final.csv")
Time_Point_10 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/T10_Final.csv")

Time_Point_1_10 <- bind_rows(Time_Point_1, Time_Point_2, Time_Point_3, Time_Point_4,
                             Time_Point_5,Time_Point_6,Time_Point_7,Time_Point_8,Time_Point_9,Time_Point_10)

Full_Data_Set <- Time_Point_1_10 %>% 
  select(cluster, N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area, Slice, X, Y, Time)

write.csv(Full_Data_Set, "D:/Membrane Experiments/P150/P150 Cc1/Outputs/Full_Data_Set.csv")

Full_Data_Set_2 <- read.csv("D:/Membrane Experiments/P150/7-17/Full_Data_Set_Processed.csv")

Combined_P150 <- bind_rows(Full_Data_Set, Full_Data_Set_2)

Combined_P150 <- Combined_P150 %>%
  filter(Time < 20 | (N_C_V2 > 0.75 & Nuclear_Membrane_BG_Subtracted > 1))

#Statistical Models
#### Duel Axis Plots #### 
#Better Statistical Models Plots 
exp_model <- nls(Combined_P150$N_C_V2 ~ A * (1 - exp(-k * Time)), 
                 data = Combined_P150, 
                 start = list(A = max(Combined_P150$N_C_V2), k = 0.1))

# Generate fitted values
Combined_P150$fit <- predict(exp_model)


#area + N/C 
max_nc <- max(Combined_P150$N_C_V2, na.rm = TRUE)
max_area <- max(Combined_P150$Nuclear_Area, na.rm = TRUE)
scale_factor <- max_nc / max_area

ggplot(Combined_P150, aes(x = Time)) +
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
max_Membrane <- max(Combined_P150$Nuclear_Membrane_BG_Subtracted, na.rm = TRUE)
max_area <- max(Combined_P150$Nuclear_Area, na.rm = TRUE)
scale_factor <- max_Membrane / max_area

ggplot(Combined_P150, aes(x = Time)) +
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
df_binned <- Combined_P150 %>%
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
    title = "Nuclear Area P150-cc1",
    x = "Time Cluster (min)",
    y = "Nuclear Area"
  ) +
  theme_minimal(base_size = 13)
