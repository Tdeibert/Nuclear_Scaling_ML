library(tidyverse)
library(broom)

Inc <- Inc %>% 
  mutate(Time = Region)
No_Inc <- No_Inc %>% 
  mutate(Time = Region)


# --- 1) Combine and bin data by 6-minute intervals ---


df <- bind_rows(
  No_Inc %>% mutate(Group = "No_Inc"),
  Inc    %>% mutate(Group = "Inc")
) %>%
  mutate(
    bin_start = floor(Time / 6) * 6,
    Time_bin_label = sprintf("%d–%d", bin_start, bin_start + 6)
  ) %>%
  arrange(bin_start) %>%
  mutate(Time_bin_label = factor(Time_bin_label, levels = unique(Time_bin_label)))

# --- 2) Plot box + points ---
p <- ggplot(df, aes(x = Time_bin_label, y = Nuclear_Area, fill = Group)) +
  geom_boxplot(
    position = position_dodge(width = 0.8),
    width = 0.7,
    outlier.shape = NA
  ) +
  # Make the jittered points black (and slightly smaller / transparent)
  geom_point(
    color = "black",
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8),
    size = 1.5,
    alpha = 0.6
  ) +
  labs(
    x = "Time (binned, minutes)",
    y = "Nuclear Area",
    title = "Comparison of Nuclear Area across 6-minute bins",
    fill = "Group"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

p

# Optional: see how many points per bin per group you have
bin_counts <- df %>%
  count(Time_bin_label, Group) %>%
  pivot_wider(names_from = Group, values_from = n, values_fill = 0) %>%
  arrange(Time_bin_label)
bin_counts

# Robust Welch t-test per bin (skip bins with too-few obs)
ttest_results <- df %>%
  filter(!is.na(Nuclear_Area)) %>%
  group_by(Time_bin_label) %>%
  group_modify(~{
    d <- .x
    # Need both groups present
    if (n_distinct(d$Group) < 2) {
      return(tibble(
        method   = "Welch Two Sample t-test",
        p.value  = NA_real_,
        note     = "Only one group present"
      ))
    }
    # Need at least 2 observations per group for stable variance
    ns <- d %>% count(Group)
    if (any(ns$n < 2)) {
      return(tibble(
        method   = "Welch Two Sample t-test",
        p.value  = NA_real_,
        note     = "Too few observations in one or both groups"
      ))
    }
    # Also guard against zero variance in a group (all identical values)
    v_by_g <- d %>% group_by(Group) %>% summarize(v = var(Nuclear_Area), .groups = "drop")
    if (any(is.na(v_by_g$v)) || any(v_by_g$v == 0)) {
      return(tibble(
        method   = "Welch Two Sample t-test",
        p.value  = NA_real_,
        note     = "Zero or undefined variance in a group"
      ))
    }
    
    # Safe t-test
    broom::tidy(t.test(Nuclear_Area ~ Group, data = d)) %>%
      select(method, p.value) %>%
      mutate(note = NA_character_)
  }) %>%
  ungroup() %>%
  mutate(
    p.adj = p.adjust(p.value, method = "BH"),
    sig = case_when(
      is.na(p.adj) ~ "NA",
      p.adj < 0.001 ~ "***",
      p.adj < 0.01  ~ "**",
      p.adj < 0.05  ~ "*",
      TRUE          ~ "ns"
    )
  )

# Rebuild significance labels (only for bins with finite p.adj)
# --- 4) Annotate plot with simplified significance labels ---
label_positions <- df %>%
  group_by(Time_bin_label) %>%
  summarize(y_pos = max(Nuclear_Area, na.rm = TRUE) * 1.05, .groups = "drop")

# Use raw p-values (not adjusted) for labeling
sig_labels <- ttest_results %>%
  mutate(label = ifelse(is.finite(p.value),
                        paste0(sig, sprintf(" (p=%.3g)", p.value)),
                        "NA")) %>%
  left_join(label_positions, by = "Time_bin_label") %>%
  filter(is.finite(p.value))

p + geom_text(
  data = sig_labels,
  aes(x = Time_bin_label, y = y_pos, label = label),
  inherit.aes = FALSE,
  size = 3.5,
  vjust = 0
)
