## ---- Libraries ----
library(tidyverse)
library(glue)
library(mgcv)        # GAM
library(minpack.lm)  # nlsLM

## ---- Load & label (unchanged) ----
Data_set_1 <- read.csv("D:/CDR2 for Analysis/CDR2/full_data_set.csv")
Data_set_2 <- Incubation_Full_Time

Data_set_1 <- Data_set_1 %>% mutate(Experiment_ID = str_glue("2025.06.25_Rep1"))
Data_set_2 <- Data_set_2 %>% rename(Time = Region) %>% mutate(Experiment_ID = str_glue("2025.06.25_Rep2"))

Combined_Data <- bind_rows(Data_set_1, Data_set_2)

Full_Data_Set <- Combined_Data %>%
  select(cluster, N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area,
         Slice, X, Y, Time, Experiment_ID) %>%
  mutate(Time = as.numeric(Time))

## ---- Strict global filtering (applies to ALL models/plots) ----
# MAD = Median Abdolute Deviation 
mad_filter <- function(x, thr = 2) {
  zrob <- (x - median(x, na.rm = TRUE)) / mad(x, constant = 1.4826, na.rm = TRUE)
  abs(zrob) <= thr
}

#IQR Filter for Data. K determines the range that is acceptable. 
iqr_keep <- function(x, k = 1) {
  q1 <- quantile(x, 0.25, na.rm = TRUE); q3 <- quantile(x, 0.75, na.rm = TRUE); i <- q3 - q1
  x >= (q1 - k*i) & x <= (q3 + k*i)
}

df_clean <- Full_Data_Set %>%
  # hard biology rules first ensures there are no negative values in the membrane chanel. 
  filter(Time <= 90, Nuclear_Membrane_BG_Subtracted > 0) %>%
  drop_na(N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area) %>%
  # strict, data-driven trimming for all three metrics
  filter(
    mad_filter(N_C_V2, thr = 2),
    mad_filter(Nuclear_Membrane_BG_Subtracted, thr = 2),
    mad_filter(Nuclear_Area, thr = 2),
    iqr_keep(N_C_V2, k = 1),
    iqr_keep(Nuclear_Membrane_BG_Subtracted, k = 1),
    iqr_keep(Nuclear_Area, k = 1),
    between(N_C_V2, quantile(N_C_V2, 0.05, na.rm = TRUE), quantile(N_C_V2, 0.95, na.rm = TRUE)),
    between(Nuclear_Membrane_BG_Subtracted, quantile(Nuclear_Membrane_BG_Subtracted, 0.05, na.rm = TRUE),
            quantile(Nuclear_Membrane_BG_Subtracted, 0.95, na.rm = TRUE)),
    between(Nuclear_Area, quantile(Nuclear_Area, 0.05, na.rm = TRUE), quantile(Nuclear_Area, 0.95, na.rm = TRUE))
  )

## ---- Common axis for all plots ----
x_scale_0_90 <- scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, by = 6), name = "Time in Minutes")

## ---- Models (all use df_clean) ----
# N/C: exponential rise to max
#Asymptotic growth model for cross sectional area and N?C ratio. 
nc_model <- nlsLM(
  N_C_V2 ~ A * (1 - exp(-k * Time)),
  data = df_clean,
  start = list(A = max(df_clean$N_C_V2, na.rm = TRUE), k = 0.01),
  control = nls.lm.control(maxiter = 500)
)

# Area: LOESS (as before)
area_loess <- loess(Nuclear_Area ~ Time, data = df_clean, span = 0.35)

# Membrane: GAM on log1p (handles dips & curvature)
df_mem <- df_clean %>% mutate(Membrane = Nuclear_Membrane_BG_Subtracted)
mem_gam <- gam(log1p(Membrane) ~ s(Time, k = 12), data = df_mem, method = "REML")

# Prediction grid
t_grid <- tibble(
  Time = seq(min(df_mem$Time, na.rm = TRUE),
             max(df_mem$Time, na.rm = TRUE),
             by = 0.5)
)

pred <- t_grid %>%
  mutate(
    nc_fit   = predict(nc_model,  newdata = t_grid),
    area_fit = pmax(0, predict(area_loess, newdata = t_grid)),
    mem_fit  = exp(predict(mem_gam, newdata = t_grid)) - 1
  )

## ---- Dual-axis scaling (99th pct to avoid outliers) ----
max_nc   <- quantile(df_clean$N_C_V2,                       0.99, na.rm = TRUE)
max_area <- quantile(df_clean$Nuclear_Area,                 0.99, na.rm = TRUE)
max_mem  <- quantile(df_clean$Nuclear_Membrane_BG_Subtracted, 0.99, na.rm = TRUE)

scale_area_on_nc  <- max_nc  / max_area
scale_area_on_mem <- max_mem / max_area

## ---- Plot 1: N/C (nls) + Area (loess) ----
ggplot() +
  geom_point(data = df_clean, aes(Time, N_C_V2, color = N_C_V2), size = 1.8, alpha = 0.6) +
  geom_line(data = pred, aes(Time, nc_fit), color = "black", linewidth = 1.1) +
  geom_point(data = df_clean, aes(Time, Nuclear_Area * scale_area_on_nc),
             color = "black", shape = 20, alpha = 0.45, size = 1.6) +
  geom_line(data = pred, aes(Time, area_fit * scale_area_on_nc),
            color = "black", linewidth = 1.0, linetype = "dashed") +
  scale_y_continuous(name = "N/C Ratio",
                     sec.axis = sec_axis(~ . / scale_area_on_nc, name = "Nuclear Area")) +
  x_scale_0_90 +
  scale_color_viridis_c(option = "magma", name = "N/C Ratio") +
  theme_minimal(base_size = 13) +
  labs(title = "N/C (nls) with Area (LOESS) — Strictly Filtered, 0–90 min")

## ---- Plot 2: Membrane (GAM) + Area (LOESS) ----
ggplot() +
  geom_point(data = df_mem, aes(Time, Membrane), color = "darkgreen", alpha = 0.45, size = 1.6) +
  geom_line(data = pred, aes(Time, mem_fit), color = "darkgreen", linewidth = 1.1) +
  geom_point(data = df_clean, aes(Time, Nuclear_Area * scale_area_on_mem),
             color = "black", shape = 20, alpha = 0.4, size = 1.6) +
  geom_line(data = pred, aes(Time, area_fit * scale_area_on_mem),
            color = "black", linewidth = 1.0) +
  scale_y_continuous(name = "Integrated Membrane Intensity",
                     sec.axis = sec_axis(~ . / scale_area_on_mem, name = "Nuclear Area")) +
  x_scale_0_90 +
  theme_minimal(base_size = 13) +
  labs(title = "Membrane (GAM) with Area (LOESS) — Strictly Filtered, 0–90 min")

#### Testing other Models#### 
mem <- df_clean %>% 
  transmute(Time, y = Nuclear_Membrane_BG_Subtracted) %>%
  arrange(Time)

# convenient lag helper: (t - t0)+
pos <- function(x) pmax(0, x)

# --- Starting values (data-driven) ---
A0 <- quantile(mem$y, 0.95, na.rm = TRUE)
t00 <- quantile(mem$Time, 0.15, na.rm = TRUE)   # rough onset time
k0  <- 0.02
lam0 <- 20
beta0 <- 1.4
Q0 <- 2; nu0 <- 1

# --- 1) Delayed exponential ---------------------------------------------------
fit_exp <- try(
  nlsLM(y ~ A * (1 - exp(-k * pos(Time - t0))),
        data = mem,
        start = list(A = A0, k = k0, t0 = t00),
        lower = c(0, 0, 0), upper = c(Inf, Inf, max(mem$Time, na.rm = TRUE)),
        control = nls.lm.control(maxiter = 1000)),
  silent = TRUE
)

# --- 2) Weibull CDF with lag --------------------------------------------------
fit_weib <- try(
  nlsLM(y ~ A * (1 - exp(- (pos(Time - t0)/lambda)^beta )),
        data = mem,
        start = list(A = A0, lambda = lam0, beta = beta0, t0 = t00),
        lower = c(0,    1e-6,   0.2, 0),
        upper = c(Inf,  Inf,    8,   max(mem$Time, na.rm = TRUE)),
        control = nls.lm.control(maxiter = 1000)),
  silent = TRUE
)

# --- 3) Richards (generalized logistic) with lag ------------------------------
fit_rich <- try(
  nlsLM(y ~ A / (1 + Q * exp(-k * pos(Time - t0)))^(1/nu),
        data = mem,
        start = list(A = A0, Q = Q0, k = k0, nu = nu0, t0 = t00),
        lower = c(0,  1e-6, 0,  0.2, 0),
        upper = c(Inf, Inf,  1,  5,   max(mem$Time, na.rm = TRUE)),
        control = nls.lm.control(maxiter = 1500)),
  silent = TRUE
)

# --- Model selection by AIC ---------------------------------------------------
#AIC = Akaike Information Criteria. Evaluates the goodness of the fit and model complexity. 
cands <- list(exp = fit_exp, weibull = fit_weib, richards = fit_rich)
ok    <- cands[ sapply(cands, function(m) inherits(m, "nls")) ]
aics  <- sapply(ok, AIC)
best_name <- names(which.min(aics))
best_fit  <- ok[[best_name]]

cat("Best membrane model by AIC:", best_name, "\n")
print(summary(best_fit))

# --- Predictions for plotting (only within observed time range) --------------
t0_obs <- min(mem$Time, na.rm = TRUE)
t1_obs <- max(mem$Time, na.rm = TRUE)
tgrid  <- tibble(Time = seq(t0_obs, t1_obs, by = 0.5))
tgrid$yhat <- predict(best_fit, newdata = tgrid)

# --- Plot membrane best-fit + (optional) 'no growth' baseline before t0_obs --
baseline <- tibble(xmin = 0, xmax = t0_obs)

ggplot(mem, aes(Time, y)) +
  geom_rect(data = baseline, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "darkgreen", alpha = 0.06) +
  geom_point(color = "darkgreen", alpha = 0.5, size = 1.6) +
  geom_line(data = tgrid, aes(Time, yhat), color = "darkgreen", linewidth = 1.1) +
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, 6), name = "Time in Minutes") +
  labs(y = "Membrane intensity (normalized)", 
       title = paste("Membrane incorporation:", best_name, "fit")) +
  theme_minimal(base_size = 13)
#### Constraining Lag#### 
mem <- df_clean %>%
  transmute(Time, y = Nuclear_Membrane_BG_Subtracted) %>%
  arrange(Time)

# 1) detect first growth time (first positive intensity after filtering)
t_start <- min(mem$Time[mem$y > 0], na.rm = TRUE)

# 2) re-index time so growth begins at t' = 0
mem <- mem %>% mutate(tprime = pmax(0, Time - t_start))

# 3) fit saturating exponential WITHOUT lag (baseline y0 allowed)
A0  <- quantile(mem$y, 0.90, na.rm = TRUE) - min(mem$y, na.rm = TRUE)
A0  <- max(A0, 1)                              # guard
y00 <- min(mem$y, na.rm = TRUE)
fit_exp0 <- nlsLM(
  y ~ y0 + A * (1 - exp(-k * tprime)),
  data   = mem,
  start  = list(y0 = y00, A = A0, k = 0.02),
  lower  = c(0,   0,  0),
  control = nls.lm.control(maxiter = 1000)
)

# (optional) a more flexible saturating curve without lag:
fit_weib0 <- nlsLM(
  y ~ y0 + A * (1 - exp(- (tprime/lambda)^beta )),
  data  = mem,
  start = list(y0 = y00, A = A0, lambda = 20, beta = 1.5),
  lower = c(0,   0, 1e-6, 0.3),
  control = nls.lm.control(maxiter = 1500)
)

# choose one (e.g., by AIC)
best_fit <- if (AIC(fit_exp0) <= AIC(fit_weib0)) fit_exp0 else fit_weib0

# predictions on observed support
tgrid <- tibble(tprime = seq(0, max(mem$tprime), by = 0.5)) %>%
  mutate(Time = tprime + t_start,
         yhat = predict(best_fit, newdata = cur_data_all()))

# plot with baseline shading for 0–t_start
ggplot(mem, aes(Time, y)) +
  geom_rect(aes(xmin = 0, xmax = t_start, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "darkgreen", alpha = 0.5) +
  geom_point(color = "darkgreen", alpha = 0.5, size = 1.6) +
  geom_line(data = tgrid, aes(Time, yhat), color = "darkgreen", linewidth = 1.1) +
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, 6),
                     name = "Time in Minutes") +
  labs(y = "Membrane intensity (normalized)",
       title = "Membrane incorporation: saturating model without lag (time re-indexed)") +
  theme_minimal(base_size = 13)

#### More Models ####
# install.packages("scam")
library(scam)

mem_sc <- mem %>% filter(Time >= t_start)
mod_sc <- scam(y ~ s(tprime, k = 12, bs = "mpi"), data = mem_sc)  # mpi = monotone increasing

tgrid_sc <- tibble(tprime = seq(0, max(mem_sc$tprime), by = 0.5),
                   Time   = tprime + t_start) %>%
  mutate(yhat = predict(mod_sc, newdata = cur_data_all()))

ggplot(mem, aes(Time, y)) +
  geom_rect(aes(xmin = 0, xmax = t_start, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "darkgreen", alpha = 0.06) +
  geom_point(color = "darkgreen", alpha = 0.5, size = 1.6) +
  geom_line(data = tgrid_sc, aes(Time, yhat), color = "darkgreen", linewidth = 1.1) +
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, 6),
                     name = "Time in Minutes") +
  labs(y = "Membrane intensity (normalized)",
       title = "Membrane incorporation: monotone smooth (SCAM) starting at first growth") +
  theme_minimal(base_size = 13)
