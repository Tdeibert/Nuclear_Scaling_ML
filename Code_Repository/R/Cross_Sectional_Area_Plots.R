#Recreating the Dynein Landing rate plot for nuclear cross sectional area data. 
library(ggplot2)
library(dplyr)

# Calculate mean per replicate per condition
replicate_means <- data %>%
  group_by(condition, replicate) %>%
  summarise(mean_area = mean(cross_sectional_area), .groups = "drop")

# Plot
ggplot(data, aes(x = condition, y = cross_sectional_area)) +
  geom_boxplot(width = 0.5, fill = "gray70", outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.6) +
  geom_point(
    data = replicate_means,
    aes(x = condition, y = mean_area),
    size = 4,
    shape = 21,
    fill = "black",
    color = "white",
    stroke = 1
  ) +
  labs(y = "Cross sectional Area", x = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank()
  )
