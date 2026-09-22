library(dplyr)
library(ggplot2)
library(boot)

# Step 1: Define prediction function
predict_nls <- function(data, indices) {
  d <- data[indices, ]
  tryCatch({
    model <- nls(N_C_V2 ~ A * (1 - exp(-k * Time)),
                 data = d,
                 start = list(A = max(d$N_C_V2), k = 0.1))
    predict(model, newdata = Full_Data_Set)
  }, error = function(e) rep(NA, nrow(Full_Data_Set)))
}

# Step 2: Run bootstrap
set.seed(123)
boot_out <- boot(data = Full_Data_Set, statistic = predict_nls, R = 200)

# Step 3: Compute 2.5% and 97.5% percentiles for confidence band
Full_Data_Set <- Full_Data_Set %>%
  mutate(fit = predict(exp_model),
         lower = apply(boot_out$t, 2, quantile, probs = 0.025, na.rm = TRUE),
         upper = apply(boot_out$t, 2, quantile, probs = 0.975, na.rm = TRUE))


ggplot(Full_Data_Set, aes(x = Time, y = N_C_V2)) +
  geom_point(aes(color = N_C_V2), size = 4) +  # Color only applied here
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey70", alpha = 0.4) +  # Fixed fill
  geom_line(aes(y = fit), color = "black", size = 1.2) +
  scale_color_viridis_c(option = "magma") +
  theme_minimal() +
  labs(
    x = "Time in Minutes",
    y = "N/C Ratio",
    title = "N/C Ratio (Exponential Plateau Model)"
  )
