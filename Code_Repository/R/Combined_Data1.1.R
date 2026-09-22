Data_set_1 <- read.csv("D:/CDR2 for Analysis/CDR2/full_data_set.csv")
Data_set_2 <- Incubation_Full_Time


Data_set_1 <- Data_set_1 %>% 
  mutate(Experiment_ID = str_glue("2025_25_06_Rep1"))
Data_set_2 <- Data_set_2 %>% 
  mutate(Experiment_ID = str_glue("2025_25_06_Rep2")) %>% 
  rename(Time = Region)

Combined_Data <- bind_rows(
  Data_set_1,
    Data_set_2
)

#Plotting Combined data

Full_Data_Set <- Combined_Data %>% 
  select(cluster, N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area, Slice, X, Y, Time, Experiment_ID)



#### Interrogating the Data for Filtering ####

Full_Data_Set <- Full_Data_Set %>%
  # remove zeros and negative membrane intensities first
  filter(Nuclear_Membrane_BG_Subtracted > 0)

Full_Data_Set %>%
  summarise(
    min_value  = min(Nuclear_Membrane_BG_Subtracted[Nuclear_Membrane_BG_Subtracted > 0], na.rm = TRUE),
    max_value  = max(Nuclear_Membrane_BG_Subtracted, na.rm = TRUE),
    mean_value = mean(Nuclear_Membrane_BG_Subtracted, na.rm = TRUE)
  )


Full_Data_Set <- Full_Data_Set %>%
  mutate(
    z_N_C_V2 = scale(N_C_V2),
    z_Nuclear_Membrane_BG_Subtracted = scale(Nuclear_Membrane_BG_Subtracted),
    z_Nuclear_Area = scale(Nuclear_Area)
  ) %>%
  filter(
    abs(z_N_C_V2) <= 3,
    abs(z_Nuclear_Membrane_BG_Subtracted) <= 3,
    abs(z_Nuclear_Area) <= 3
  )


# Helper: IQR filter for one column
filter_iqr <- function(df, column, k = 1.5) {
  q1  <- quantile(df[[column]], 0.25, na.rm = TRUE)
  q3  <- quantile(df[[column]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  df %>%
    filter(
      !!sym(column) >= (q1 - k * iqr),
      !!sym(column) <= (q3 + k * iqr)
    )
}

# 1) Enforce membrane rule (strict): remove zero/negative values ONLY for membrane
# 2) Then apply IQR-based outlier filtering on the three columns
Filtered <- Full_Data_Set %>%
  filter(Nuclear_Membrane_BG_Subtracted > 0) %>%
  filter_iqr("N_C_V2", k = 1.5) %>%
  filter_iqr("Nuclear_Membrane_BG_Subtracted", k = 1.5) %>%
  filter_iqr("Nuclear_Area", k = 1.5)


Filtered <- Full_Data_Set %>%
  filter(Nuclear_Membrane_BG_Subtracted > 0) %>%
  mutate(
    zrob_N_C_V2  = (N_C_V2 - median(N_C_V2, na.rm = TRUE)) /
      mad(N_C_V2,  constant = 1.4826, na.rm = TRUE),
    zrob_Membr   = (Nuclear_Membrane_BG_Subtracted - median(Nuclear_Membrane_BG_Subtracted, na.rm = TRUE)) /
      mad(Nuclear_Membrane_BG_Subtracted, constant = 1.4826, na.rm = TRUE),
    zrob_Area    = (Nuclear_Area - median(Nuclear_Area, na.rm = TRUE)) /
      mad(Nuclear_Area, constant = 1.4826, na.rm = TRUE)
  ) %>%
  filter(
    abs(zrob_N_C_V2)  <= 3,
    abs(zrob_Membr)   <= 3,
    abs(zrob_Area)    <= 3
  ) %>%
  select(-starts_with("zrob_"))



#### Generating Plots of the Data #### 


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



####Statistical Modeling####
#Better Statistical Models Plots 
#Area Model
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

#Statistical Models

#Nuclear Area Model Fit 
exp_model <- nls(Full_Data_Set$N_C_V2 ~ A * (1 - exp(-k * Time)), 
                 data = Full_Data_Set, 
                 start = list(A = max(Full_Data_Set$N_C_V2), k = 0.1))

#Membrane Fit 
mem_model <- nls(
  Nuclear_Membrane_BG_Subtracted ~ A * (1 - exp(-k * Time)),
  data = Full_Data_Set,
  start = list(
    A = max(Full_Data_Set$Nuclear_Membrane_BG_Subtracted, na.rm = TRUE),
    k = 0.001    # smaller rate constant to start
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)

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




#Nuclear Area + Membrane

# I want to use the statistical model from the area + N/C Plot for the Area axis only. I want to use the new membrane model we just generated for the membrane axis. 

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


summary(Full_Data_Set$Nuclear_Membrane_BG_Subtracted)


# Fit an exponential rise-to-maximum model for the membrane intensity
mem_model <- nls(
  Nuclear_Membrane_BG_Subtracted ~ A * (1 - exp(-k * Time)),
  data = Full_Data_Set,
  start = list(
    A = max(Full_Data_Set$Nuclear_Membrane_BG_Subtracted, na.rm = TRUE),
    k = 0.001    # smaller rate constant to start
  ),
  control = nls.control(maxiter = 500, warnOnly = TRUE)
)

# Add fitted values back into the data frame
Full_Data_Set$mem_fit <- predict(mem_model)

# Scaling factor between membrane and nuclear area for dual-axis plotting
max_mem <- max(Full_Data_Set$Nuclear_Membrane_BG_Subtracted, na.rm = TRUE)
max_area <- max(Full_Data_Set$Nuclear_Area, na.rm = TRUE)
scale_factor <- max_mem / max_area

# Plot with exponential membrane model + LOESS for area
ggplot(Full_Data_Set, aes(x = Time)) +
  # Primary Y-axis: membrane intensity
  geom_point(aes(y = Nuclear_Membrane_BG_Subtracted),
             color = "darkgreen", alpha = 0.6, size = 1.8) +
  geom_line(aes(y = mem_fit),
            color = "black", size = 1.2) +
  
  # Secondary Y-axis: nuclear area (scaled)
  geom_point(aes(y = Nuclear_Area * scale_factor),
             shape = 20, color = "black", alpha = 0.6, size = 1.8) +
  geom_smooth(aes(y = Nuclear_Area * scale_factor),
              method = "loess", se = FALSE,
              color = "black", linetype = "solid", size = 1.2, span = 0.3) +
  
  # Axis definitions
  scale_y_continuous(
    name = "Integrated Membrane Intensity (Modeled)",
    sec.axis = sec_axis(~ . / scale_factor, name = "Nuclear Area")
  ) +
  scale_x_continuous(
    breaks = seq(0, max(Full_Data_Set$Time, na.rm = TRUE), by = 6),
    name = "Time in Minutes"
  ) +
  
  theme_minimal(base_size = 13) +
  labs(
    title = "Exponential Fit of Membrane Intensity with LOESS Area Trend"
  )


# 2) Fit GAM on log1p to stabilize variance (handles wide dynamic range)
mem_gam <- gam(log1p(Membrane) ~ s(Time, k = 12),         # k ~ flexibility; 10–20 is typical
               data = df, method = "REML")

# 3) Back-transform predictions
df$mem_fit <- exp(predict(mem_gam, newdata = df)) - 1

# 4) Dual-axis scaling (use 99th pct to avoid outlier domination)
max_mem  <- quantile(df$Membrane, 0.99, na.rm = TRUE)
max_area <- quantile(df$Nuclear_Area, 0.99, na.rm = TRUE)
scale_factor <- max_mem / max_area

# 5) Plot: GAM membrane fit + LOESS for area
ggplot(df, aes(x = Time)) +
  # Membrane points + GAM fit (primary axis)
  geom_point(aes(y = Membrane), color = "darkgreen", alpha = 0.5, size = 1.6) +
  geom_line(aes(y = mem_fit), color = "darkgreen", linewidth = 1.1) +
  
  # Area points + LOESS on secondary axis
  geom_point(aes(y = Nuclear_Area * scale_factor),
             color = "black", alpha = 0.4, size = 1.6, shape = 20) +
  geom_smooth(aes(y = Nuclear_Area * scale_factor),
              method = "loess", se = FALSE, span = 0.35,
              color = "black", linewidth = 1) +
  
  scale_y_continuous(
    name = "Integrated Membrane Intensity",
    sec.axis = sec_axis(~ . / scale_factor, name = "Nuclear Area")
  ) +
  scale_x_continuous(breaks = seq(0, max(df$Time, na.rm = TRUE), by = 6),
                     name = "Time in Minutes") +
  labs(title = "Membrane Intensity (GAM fit) with LOESS Area Trend") +
  theme_minimal(base_size = 13)
