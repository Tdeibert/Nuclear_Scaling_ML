#Exploring the possiblities of differences in growth velocites per condition
library(tidyverse)
library(broom)
library(minpack.lm)  # for nlsLM
library(segmented)    # for piecewise linear model


#Data Imports

Control <- read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/Control/Time_Point_1_10.csv")

P150 <- read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/P150/4_13/Data_Combined.csv")

CDR2_1 <- read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/CDR2/Data_Sets/full_data_set_5_29.csv")

CDR2_2 <-read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/CDR2/Data_Sets/Incubation_Full_100_Min_02_25_2025.csv")

#Data Cleaning 
Control <- Control %>%  
  rename(Time = Region)

CDR2 <- (bind_rows(CDR2_1, CDR2_2))


#### Exponential Plateau Model (robust version) ####

fit_exp_plateau <- function(df, time_col = "Time", area_col = "Area") {
  
  df2 <- df %>%
    dplyr::filter(
      !is.na(.data[[time_col]]),
      !is.na(.data[[area_col]])
    ) %>%
    dplyr::mutate(
      .Time = .data[[time_col]],
      .Area = .data[[area_col]]
    )
  
  start_Amax <- max(df2$.Area, na.rm = TRUE)
  start_k    <- 0.05
  
  nlsLM(
    .Area ~ Amax * (1 - exp(-k * .Time)),
    data   = df2,
    start  = list(Amax = start_Amax, k = start_k),
    control = nls.lm.control(maxiter = 500)
  )
}

summarise_exp_plateau <- function(fit) {
  co <- coef(fit)   # named numeric vector
  
  Amax <- unname(co["Amax"])
  k    <- unname(co["k"])
  
  tibble::tibble(
    Amax   = Amax,
    k      = k,
    v0     = Amax * k,       # initial growth velocity
    t_half = log(2) / k      # time to 50% of Amax
  )
}


Control <- Control %>%
  dplyr::mutate(Condition = "Control")

P150 <- P150 %>%
  dplyr::mutate(Condition = "P150")

CDR2 <- CDR2 %>%
  dplyr::mutate(Condition = "CDR2")

Control <- Control[, c("Condition", "Time", "Nuclear_Area")]
P150    <- P150[,    c("Condition", "Time", "Nuclear_Area")]
CDR2    <- CDR2[,    c("Condition", "Time", "Nuclear_Area")]

nuc_data <- dplyr::bind_rows(Control, P150, CDR2) %>%
  dplyr::rename(Area = Nuclear_Area)


str(nuc_data)
# Should show: Condition (chr), Time (num), Area (num)


control_df <- dplyr::filter(nuc_data, Condition == "Control")
p150_df    <- dplyr::filter(nuc_data, Condition == "P150")
cdr2_df    <- dplyr::filter(nuc_data, Condition == "CDR2")

fit_control <- fit_exp_plateau(control_df, time_col = "Time", area_col = "Area")
fit_p150    <- fit_exp_plateau(p150_df,    time_col = "Time", area_col = "Area")
fit_cdr2    <- fit_exp_plateau(cdr2_df,    time_col = "Time", area_col = "Area")

summarise_exp_plateau(fit_control)
summarise_exp_plateau(fit_p150)
summarise_exp_plateau(fit_cdr2)

fits_list <- nuc_data %>%
  split(.$Condition) %>%              # list: one df per condition
  lapply(function(df_cond) {
    cond <- unique(df_cond$Condition)
    fit  <- fit_exp_plateau(df_cond, time_col = "Time", area_col = "Area")
    params <- summarise_exp_plateau(fit)
    dplyr::mutate(params, Condition = cond)
  })

exp_fits <- dplyr::bind_rows(fits_list)

exp_fits

predict_exp_plateau <- function(fit, t_vec) {
  co <- coef(fit)
  Amax <- unname(co["Amax"])
  k    <- unname(co["k"])
  
  tibble::tibble(
    Time     = t_vec,
    Area_fit = Amax * (1 - exp(-k * Time)),
    v_dt     = Amax * k * exp(-k * Time)
  )
}

time_grid <- seq(min(nuc_data$Time), max(nuc_data$Time), by = 0.5)

exp_curves <- nuc_data %>%
  dplyr::group_by(Condition) %>%
  dplyr::group_modify(~ {
    fit   <- fit_exp_plateau(.x, time_col = "Time", area_col = "Area")
    preds <- predict_exp_plateau(fit, time_grid)
    preds               # <-- NO Condition column here
  })


ggplot(nuc_data, aes(Time, colour = Condition)) +
  geom_point(aes(y = Area), alpha = 0.3) +
  geom_line(
    data = exp_curves,
    aes(y = Area_fit, group = Condition),
    linewidth = 1
  ) +
  theme_classic()

library(dplyr)
library(ggplot2)

# build model curves separately for each condition
split_df <- split(nuc_data, nuc_data$Condition)

curve_list <- lapply(names(split_df), function(cond) {
  dfc <- split_df[[cond]]
  tg  <- seq(0, max(dfc$Time), by = 1)      # prediction grid per condition
  fit <- fit_exp_plateau(dfc, time_col = "Time", area_col = "Area")
  
  predict_exp_plateau(fit, tg) %>%
    mutate(Condition = cond)
})

exp_curves <- bind_rows(curve_list)

ggplot() +
  geom_point(data = nuc_data,
             aes(x = Time, y = Area),
             alpha = 0.2, size = 0.7) +
  geom_line(data = exp_curves,
            aes(x = Time, y = Area_fit),
            colour = "black", linewidth = 1) +
  facet_wrap(~ Condition, ncol = 1, scales = "fixed") +
  theme_classic() +
  labs(x = "Time (min)",
       y = "Nuclear cross-sectional area",
       title = "Nuclear growth curves with exponential plateau fits")

# here I use 6-min bins; change divisor if you prefer 10, etc.
nuc_binned <- nuc_data %>%
  mutate(Time_bin = floor(Time / 6) * 6) %>%
  group_by(Condition, Time_bin) %>%
  summarise(
    mean_area = mean(Area, na.rm = TRUE),
    se_area   = sd(Area, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

ggplot(nuc_binned,
       aes(x = Time_bin, y = mean_area, colour = Condition)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = mean_area - se_area,
                    ymax = mean_area + se_area),
                width = 2, alpha = 0.6) +
  theme_classic() +
  labs(x = "Time (min, binned)",
       y = "Mean nuclear area ± SE",
       title = "Average nuclear growth over time")

vel_curves <- exp_curves %>%
  dplyr::select(Time, v_dt, Condition)


ggplot(vel_curves,
       aes(x = Time, y = v_dt, colour = Condition)) +
  geom_line(linewidth = 1) +
  theme_classic() +
  labs(x = "Time (min)",
       y = "Growth velocity (ΔArea / min)",
       title = "Modeled nuclear growth velocity over time")


####T1/2 ####
summary_tbl <- nuc_data %>%
  dplyr::group_by(Condition) %>%
  dplyr::group_modify(~{
    fit <- fit_exp_plateau(.x, time_col = "Time", area_col = "Area")
    summarise_exp_plateau(fit)
  })

summary_tbl


ggplot(summary_tbl, aes(x = Condition, y = t_half, fill = Condition)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_text(aes(label = round(t_half, 1)),
            vjust = -0.5, size = 4) +
  theme_classic() +
  labs(
    title = "Half-time to Reach 50% of Nuclear Growth Plateau",
    x = "",
    y = "t₁/₂ (minutes)"
  ) +
  theme(legend.position = "none")

ggplot(summary_tbl, aes(Condition, t_half, colour = Condition)) +
  geom_point(size = 4) +
  theme_classic() +
  labs(
    title = "Nuclear Growth Half-time (t₁/₂)",
    x = "",
    y = "Half-time (minutes)"
  ) +
  theme(legend.position = "none")

summary_tbl <- summary_tbl %>%
  mutate(v_half = v0 / 2)


ggplot(summary_tbl, aes(Condition, v_half, fill = Condition)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_text(aes(label = round(v_half, 2)),
            vjust = -0.5, size = 4) +
  theme_classic() +
  labs(
    title = "Growth Velocity at Half-time (v₁/₂)",
    x = "",
    y = "Velocity at t₁/₂ (Area units/min)"
  ) +
  theme(legend.position = "none")

summary_tbl %>%
  dplyr::select(Condition, Amax, k, v0, t_half, v_half)
