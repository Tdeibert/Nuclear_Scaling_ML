library(tidyverse)
damage_data <- read_csv("./SIA 24 Hours.csv")
damage_data <- damage_data %>%
  mutate(
    Treatment = as.character(Treatment),             # treat as raw character first
    Treatment = str_trim(Treatment),                 # remove any extra spaces
    Treatment = ifelse(Treatment == "ICU", "IUC", Treatment),  # fix ICU → IUC
    Prefix = str_extract(Treatment, "^[A-Z]+")       # extract prefix *before* factor conversion
  ) %>%
  filter(!is.na(Treatment)) %>%                      # remove actual NA rows
  mutate(
    Treatment = factor(Treatment, levels = c(        # now safely convert to factor
      "AL0.1", "AL1", "GAL1", "GAL10", "GAL20", 
      "CUR10", "CUR50", "CUR100", 
      "CAR9.6", "CAR19.8", 
      "MON", "TOL", "IUC"
    ))
  )

# Step 2: Compute summary for placing significance stars
damage_summary <- damage_data %>%
  group_by(Treatment) %>%
  summarise(y_pos = max(`Cell Damage`) + 0.3, .groups = "drop")

# Step 3: Add significance levels manually
significance_labels <- tribble(
  ~Treatment, ~Significance,
  "CUR10", "***",
  "CUR50", "**",
  "CUR100", "***",
  "MON", "**"
)

# Join with summary for plotting text
label_df <- left_join(damage_summary, significance_labels, by = "Treatment")

# Step 4: Define color map for prefixes
prefix_colors <- c(
  "AL" = "#E69F00",
  "GAL" = "#56B4E9",
  "CUR" = "#009E73",
  "CAR" = "#D55E00",
  "MON" = "#CC79A7",
  "TOL" = "#F0E442",
  "IUC" = "#000000"
)

ggplot(damage_data, aes(x = Treatment, y = `Cell Damage`, fill = Prefix)) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_point(aes(color = Prefix),
             position = position_nudge(x = 0), 
             size = 1.5, alpha = 0.7, shape = 16) +
  geom_text(data = label_df,
            aes(x = Treatment, y = y_pos, label = Significance),
            inherit.aes = FALSE,
            color = "red",
            size = 5) +
  scale_fill_manual(values = prefix_colors, name = "Treatment Group") +
  scale_color_manual(values = prefix_colors, guide = "none") +
  labs(
    title = "SIA Cell Damage 24 hpi",
    x = "Treatments",
    y = "Cell Damage Score"
  ) +
  scale_y_continuous(limits = c(0, 5.2)) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right",
    plot.title = element_text(hjust = 0.5)
  )
