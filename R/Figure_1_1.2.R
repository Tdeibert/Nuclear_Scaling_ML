###############################################
## Nuclear Scaling – Area / N:C Comparisons  ##
## Cleaned & Annotated Script                ##
###############################################

## ---- 0. Libraries & Global Settings -----------------------------------------
library(tidyverse)
library(ggplot2)
library(readxl)
library(ggpubr)
library(rstatix)
library(gt)
library(paletteer)

## ---- 1. Global Color Constants & Helpers ------------------------------------

# Main color palette
Control <- "#117733"
P150    <- "#88CCEE"
CDR2    <- "#882255"
N_C     <- "black"

# Dual-axis scaling factor:
#   N/C ratio is 0–1; we want the right axis to span ~0–500 μm^2.
scale_factor <- 500

# Helper: bin time into 6-min intervals
bin_time_6min <- function(df, time_col = "Time") {
  df %>%
    mutate(
      Time_Binned = floor(.data[[time_col]] / 6) * 6
    )
}

# Helper: single-condition scatter + LOESS for Nuclear_Area
plot_single_area <- function(df, col, title = NULL) {
  ggplot(df, aes(x = Time, y = Nuclear_Area)) +
    geom_point(size = 2, color = "black", fill = col,
               alpha = 0.8, shape = 21) +
    geom_smooth(method = "loess", color = "black") +
    scale_x_continuous(limits = c(0, 90),
                       breaks = seq(0, 90, 10)) +
    scale_y_continuous(limits = c(0, 500),
                       breaks = seq(0, 500, 100)) +
    labs(
      y = "nuclear cross-sectional area (µm^2)",
      x = "time (minutes)",
      title = title
    ) +
    theme_classic(base_size = 14)
}

# Helper: single-condition scatter + LOESS for N/C ratio
plot_single_nc <- function(df, title = NULL) {
  ggplot(df, aes(x = Time, y = N_C_V2)) +
    geom_point(size = 2, color = "black", fill = N_C,
               alpha = 0.5, shape = 21) +
    geom_smooth(method = "loess", color = "black") +
    scale_x_continuous(limits = c(0, 90),
                       breaks = seq(0, 90, 10)) +
    scale_y_continuous(limits = c(0, 1),
                       breaks = seq(0, 1, 0.1)) +
    labs(
      y = "N/C ratio",
      x = "time (minutes)",
      title = title
    ) +
    theme_classic(base_size = 14)
}

# Helper: dual-axis plot for one condition
plot_dual_axis <- function(df, area_col, title = NULL) {
  ggplot(df, aes(x = Time)) +
    # Left axis: N/C ratio
    geom_point(aes(y = N_C_V2), size = 2, color = "black",
               fill = N_C, alpha = 0.5, shape = 21) +
    geom_smooth(aes(y = N_C_V2), method = "loess", color = N_C) +
    
    # Right axis: Nuclear Area scaled by scale_factor
    geom_point(aes(y = Nuclear_Area / scale_factor),
               size = 2, color = "black",
               fill = area_col, alpha = 0.8, shape = 21) +
    geom_smooth(aes(y = Nuclear_Area / scale_factor),
                method = "loess", color = area_col) +
    
    scale_x_continuous(limits = c(0, 90),
                       breaks = seq(0, 90, 10)) +
    scale_y_continuous(
      name   = "N/C ratio",
      limits = c(0, 1),
      breaks = seq(0, 1, 0.1),
      sec.axis = sec_axis(~ . * scale_factor,
                          name = "nuclear cross-sectional area (µm^2)")
    ) +
    labs(
      x     = "time (minutes)",
      title = title
    ) +
    theme_classic(base_size = 14) +
    theme(
      axis.title.y.left  = element_text(color = N_C),
      axis.text.y.left   = element_text(color = N_C),
      axis.title.y.right = element_text(color = "black"),
      axis.text.y.right  = element_text(color = "black")
    )
}

# Helper: single-condition binned boxplot
plot_single_box <- function(binned_df, col, title = NULL) {
  ggplot(binned_df, aes(x = Time_Binned, y = Nuclear_Area,
                        group = Time_Binned)) +
    geom_boxplot(
      width = 3,
      alpha = 0.8,
      color = "black",
      fill  = col,
      outlier.shape = NA
    ) +
    geom_jitter(width = 1.5, alpha = 0.3,
                color = col, size = 1) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, 6),
      name   = "time (minutes)"
    ) +
    scale_y_continuous(
      limits = c(0, 500),
      breaks = seq(0, 500, 100),
      name   = "nuclear cross-sectional area (µm^2)"
    ) +
    labs(title = title) +
    theme_classic(base_size = 14)
}

## ---- 2. Load & Preprocess Each Condition -----------------------------------

### 2.1 Control ---------------------------------------------------------------

Control_Data <- read.csv(
  "C:/Users/tdeibert/.Working_Docs_Folder/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_3/Time_Point_1_10.csv"
)

# In this file, "Region" encodes time
Control_Data <- Control_Data %>%
  mutate(Time = Region)

Control_Binned <- Control_Data %>%
  bin_time_6min("Time")


# Control Plots:
 plot_single_area(Control_Data, Control, "Control – Nuclear Area")
 plot_single_nc(Control_Data, "Control – N/C ratio")
 plot_dual_axis(Control_Data, Control, "Control – Dual Axis")
 plot_single_box(Control_Binned, Control, "Control – Binned Area")


### 2.2 P150 -------------------------------------------------------------------

P150_Data <- read.csv(
  "C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/P150/p150_combined.csv"
)

P150_filtered <- P150_Data %>%
  # (Comment says “after 30 min”, filter uses 15; adjust if needed)
  filter(Time < 15 | Nuclear_Area >= 50) %>%                    # remove small nuclei early
  filter(                                                         # ±2 SD filter
    Nuclear_Area >= mean(Nuclear_Area, na.rm = TRUE) - 2 * sd(Nuclear_Area, na.rm = TRUE),
    Nuclear_Area <= mean(Nuclear_Area, na.rm = TRUE) + 2 * sd(Nuclear_Area, na.rm = TRUE)
  )

P150_Binned <- P150_filtered %>%
  bin_time_6min("Time")

# P150 plots:
plot_single_area(P150_filtered, P150, "p150-CC1 – Nuclear Area")
plot_single_nc(P150_filtered, "p150-CC1 – N/C ratio")
plot_dual_axis(P150_filtered, P150, "p150-CC1 – Dual Axis")
plot_single_box(P150_Binned, P150, "p150-CC1 – Binned Area")


### 2.3 CDR2 -------------------------------------------------------------------

CDR2_6_25 <- read.csv(
  "C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/CDR2/Data_Sets/full_data_set_5_29.csv"
)
CDR2_9_27 <- read.csv(
  "C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/CDR2/Data_Sets/Incubation_Full_100_Min_02_25_2025.csv"
)

CDR2_Data <- bind_rows(CDR2_6_25, CDR2_9_27)

CDR2_Binned <- CDR2_Data %>%
  bin_time_6min("Time")

# CDR2 plots:
plot_single_area(CDR2_Data, CDR2, "CDR2 – Nuclear Area")
plot_single_nc(CDR2_Data, "CDR2 – N/C ratio")
plot_dual_axis(CDR2_Data, CDR2, "CDR2 – Dual Axis")
plot_single_box(CDR2_Binned, CDR2, "CDR2 – Binned Area")


## ---- 3. Combine Binned Data for Comparisons --------------------------------

Control_Binned <- Control_Binned %>% mutate(Condition = "Control")
P150_Binned    <- P150_Binned    %>% mutate(Condition = "P150")
CDR2_Binned    <- CDR2_Binned    %>% mutate(Condition = "CDR2")

Combined_Binned <- bind_rows(
  Control_Binned %>% select(-any_of("X.1")),
  P150_Binned    %>% select(-any_of("X.1")),
  CDR2_Binned    %>% select(-any_of("X.1"))
) %>%
  mutate(
    Time_Binned = factor(Time_Binned,
                         levels = sort(unique(Time_Binned))),
    Condition   = factor(Condition,
                         levels = c("Control", "P150", "CDR2"))
  )

bin_levels <- levels(Combined_Binned$Time_Binned)


## ---- 4. Per-Bin Pairwise Tests (Control vs P150 / CDR2) --------------------

# Helper: per-bin pairwise test
pairwise_bin_test <- function(dat, cond_a, cond_b,
                              method = c("t.test", "wilcox.test")[1]) {
  dat %>%
    filter(Condition %in% c(cond_a, cond_b)) %>%
    group_by(Time_Binned) %>%
    summarise(
      p = {
        x <- Nuclear_Area[Condition == cond_a]
        y <- Nuclear_Area[Condition == cond_b]
        if (length(x) > 1 && length(y) > 1) {
          if (method == "t.test") t.test(x, y)$p.value
          else wilcox.test(x, y)$p.value
        } else NA_real_
      },
      .groups = "drop"
    ) %>%
    mutate(
      group1   = cond_a,
      group2   = cond_b,
      p.signif = case_when(
        is.na(p)  ~ "ns",
        p < 1e-4  ~ "****",
        p < 1e-3  ~ "***",
        p < 1e-2  ~ "**",
        p < 0.05  ~ "*",
        TRUE      ~ "ns"
      )
    )
}

# Build both comparisons
p_P150 <- pairwise_bin_test(Combined_Binned, "Control", "P150",
                            method = "t.test")
p_CDR2 <- pairwise_bin_test(Combined_Binned, "Control", "CDR2",
                            method = "t.test")

# Attach y-positions for stars / brackets
p_all <- bind_rows(p_P150, p_CDR2) %>%
  filter(!is.na(Time_Binned)) %>%
  left_join(
    Combined_Binned %>%
      group_by(Time_Binned) %>%
      summarise(ymax = max(Nuclear_Area, na.rm = TRUE),
                .groups = "drop"),
    by = "Time_Binned"
  ) %>%
  mutate(
    Time_Binned = factor(Time_Binned, levels = bin_levels),
    
    # base position just above tallest box
    y.position  = ymax + 12,
    
    # put P150 comparison higher; CDR2 closer to boxes
    y.position  = if_else(group2 == "P150", y.position + 22, y.position),
    
    # for completeness
    p_label          = sprintf("p = %.2g", p),
    y.position.label = y.position + 15
  )


## ---- 5. Bracket Geometry for Custom Colored Stars --------------------------

p_brackets <- p_all %>%
  filter(p.signif != "ns") %>%     # drop non-significant
  mutate(
    x_center = as.numeric(Time_Binned),
    
    group_index1 = case_when(
      group1 == "Control" ~ 1L,
      group1 == "P150"    ~ 2L,
      group1 == "CDR2"    ~ 3L
    ),
    group_index2 = case_when(
      group2 == "Control" ~ 1L,
      group2 == "P150"    ~ 2L,
      group2 == "CDR2"    ~ 3L
    ),
    
    # color by the non-Control group (matches bar color)
    star_color = case_when(
      group2 == "P150" ~ P150,
      group2 == "CDR2" ~ CDR2
    ),
    
    n_groups = 3L,
    spacing  = 0.75 / n_groups,         # box dodge width
    center_i = (n_groups + 1L) / 2L,
    
    offset1 = (group_index1 - center_i) * spacing,
    offset2 = (group_index2 - center_i) * spacing,
    
    x1   = x_center + offset1,
    x2   = x_center + offset2,
    xmid = (x1 + x2) / 2
  )


## ---- 6. Full-Time Boxplot with Colored Stars/Brackets ----------------------

plot_full <- ggplot(Combined_Binned,
                    aes(x = Time_Binned,
                        y = Nuclear_Area,
                        fill = Condition)) +
  geom_boxplot(
    position     = position_dodge2(width = 0.75, preserve = "single"),
    width        = 0.65,
    alpha        = 0.85,
    color        = "black",
    outlier.shape = NA
  ) +
  # Brackets
  geom_segment(
    data = p_brackets,
    aes(x = x1, xend = x2, y = y.position, yend = y.position),
    inherit.aes = FALSE,
    linewidth   = 0.6
  ) +
  geom_segment(
    data = p_brackets,
    aes(x = x1, xend = x1, y = y.position, yend = y.position - 5),
    inherit.aes = FALSE,
    linewidth   = 0.6
  ) +
  geom_segment(
    data = p_brackets,
    aes(x = x2, xend = x2, y = y.position, yend = y.position - 5),
    inherit.aes = FALSE,
    linewidth   = 0.6
  ) +
  # Stars
  geom_text(
    data = p_brackets,
    aes(x = xmid, y = y.position + 5,
        label = p.signif, color = star_color),
    inherit.aes = FALSE,
    size        = 5,
    fontface    = "bold"
  ) +
  scale_color_identity(guide = "none") +  # don't show star_color legend
  scale_y_continuous(limits = c(0, 600),
                     breaks = seq(0, 600, 100)) +
  scale_fill_manual(
    values = c(
      "Control" = Control,
      "P150"    = P150,
      "CDR2"    = CDR2
    ),
    breaks = c("Control", "P150", "CDR2"),
    labels = c("Control", "p150-CC1", "CDR2"),
    name   = NULL
  ) +
  labs(
    x    = "time binned (min)",
    y    = "nuclear cross-sectional area (µm^2)",
    fill = NULL
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  ) +
  coord_cartesian(clip = "off")

plot_full


## ---- 7. Early-Time Only Plot (6–60 min) ------------------------------------

early_bins <- c("6", "12", "18", "24", "30", "36", "42", "48", "54", "60")

Combined_early <- Combined_Binned %>%
  filter(Time_Binned %in% early_bins) %>%
  droplevels()

p_all_early <- p_all %>%
  filter(Time_Binned %in% early_bins) %>%
  mutate(Time_Binned = droplevels(Time_Binned))

p_brackets_early <- p_all_early %>%
  filter(p.signif != "ns") %>%
  mutate(
    x_center = as.numeric(Time_Binned),
    
    group_index1 = case_when(
      group1 == "Control" ~ 1L,
      group1 == "P150"    ~ 2L,
      group1 == "CDR2"    ~ 3L
    ),
    group_index2 = case_when(
      group2 == "Control" ~ 1L,
      group2 == "P150"    ~ 2L,
      group2 == "CDR2"    ~ 3L
    ),
    
    star_color = case_when(
      group2 == "P150" ~ P150,
      group2 == "CDR2" ~ CDR2
    ),
    
    n_groups = 3L,
    spacing  = 0.75 / n_groups,
    center_i = (n_groups + 1L) / 2L,
    
    offset1 = (group_index1 - center_i) * spacing,
    offset2 = (group_index2 - center_i) * spacing,
    
    x1   = x_center + offset1,
    x2   = x_center + offset2,
    xmid = (x1 + x2) / 2
  )

plot_early <- ggplot(Combined_early,
                     aes(x = Time_Binned,
                         y = Nuclear_Area,
                         fill = Condition)) +
  geom_boxplot(
    position     = position_dodge2(width = 0.75, preserve = "single"),
    width        = 0.65,
    alpha        = 0.85,
    color        = "black",
    outlier.shape = NA
  ) +
  geom_segment(
    data = p_brackets_early,
    aes(x = x1, xend = x2, y = y.position, yend = y.position,
        color = star_color),
    inherit.aes = FALSE,
    linewidth   = 0.7
  ) +
  geom_segment(
    data = p_brackets_early,
    aes(x = x1, xend = x1, y = y.position, yend = y.position - 5,
        color = star_color),
    inherit.aes = FALSE,
    linewidth   = 0.7
  ) +
  geom_segment(
    data = p_brackets_early,
    aes(x = x2, xend = x2, y = y.position, yend = y.position - 5,
        color = star_color),
    inherit.aes = FALSE,
    linewidth   = 0.7
  ) +
  geom_text(
    data = p_brackets_early,
    aes(x = xmid, y = y.position + 5, label = p.signif,
        color = star_color),
    inherit.aes = FALSE,
    size        = 5,
    fontface    = "bold"
  ) +
  scale_color_identity(guide = "none") +
  scale_fill_manual(
    values = c(
      "Control" = Control,
      "P150"    = P150,
      "CDR2"    = CDR2
    ),
    breaks = c("Control", "P150", "CDR2"),
    labels = c("Control", "p150-CC1", "CDR2"),
    name   = NULL
  ) +
  scale_y_continuous(limits = c(0, 600),
                     breaks = seq(0, 600, 100)) +
  labs(
    x    = "time binned (min)",
    y    = "nuclear cross-sectional area (µm^2)",
    fill = NULL
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  ) +
  coord_cartesian(clip = "off")

plot_early


## ---- 8. Per-Bin Summary Table (gt) -----------------------------------------

test_method <- "t.test"  # or "wilcox.test"
digits_p    <- 3

# Helper: run one two-sample test inside a bin
bin_test_once <- function(df, cond_a, cond_b,
                          method = c("t.test", "wilcox.test")[1]) {
  x <- df$Nuclear_Area[df$Condition == cond_a]
  y <- df$Nuclear_Area[df$Condition == cond_b]
  
  if (length(x) < 2 || length(y) < 2) {
    tibble(
      p      = NA_real_,
      eff    = NA_real_,
      mean_a = mean(x),
      mean_b = mean(y),
      n_a    = length(x),
      n_b    = length(y)
    )
  } else if (method == "t.test") {
    tt <- t.test(x, y)
    tibble(
      p      = tt$p.value,
      eff    = mean(y) - mean(x),  # mean difference
      mean_a = mean(x),
      mean_b = mean(y),
      n_a    = length(x),
      n_b    = length(y)
    )
  } else {
    ww <- wilcox.test(x, y, exact = FALSE)
    tibble(
      p      = ww$p.value,
      eff    = median(y) - median(x),
      mean_a = mean(x),
      mean_b = mean(y),
      n_a    = length(x),
      n_b    = length(y)
    )
  }
}

# Define contrasts of interest
contrasts <- tribble(
  ~cond_a,   ~cond_b,   ~contrast,
  "Control", "P150",    "P150 vs Control",
  "Control", "CDR2",    "CDR2 vs Control"
)

# Long-format table with per-bin stats
p_long <- Combined_Binned %>%
  group_by(Time_Binned) %>%
  group_modify(~ {
    dat <- .x
    contrasts %>%
      mutate(
        res = pmap(
          list(cond_a, cond_b),
          ~ bin_test_once(
            dat %>% filter(Condition %in% c(..1, ..2)),
            ..1, ..2, method = test_method
          )
        )
      ) %>%
      unnest(res)
  }) %>%
  ungroup() %>%
  group_by(contrast) %>%
  mutate(p_adj_FDR = p.adjust(p, method = "BH")) %>%
  ungroup() %>%
  mutate(
    p_fmt     = ifelse(is.na(p), NA, formatC(p, format = "e", digits = digits_p)),
    p_adj_fmt = ifelse(is.na(p_adj_FDR), NA,
                       formatC(p_adj_FDR, format = "e", digits = digits_p)),
    signif    = case_when(
      is.na(p_adj_FDR) ~ "ns",
      p_adj_FDR < 1e-4 ~ "****",
      p_adj_FDR < 1e-3 ~ "***",
      p_adj_FDR < 1e-2 ~ "**",
      p_adj_FDR < 0.05 ~ "*",
      TRUE             ~ "ns"
    )
  )

# Wide view: one row per time bin
p_wide <- p_long %>%
  select(Time_Binned, contrast,
         p = p_fmt, p_adj = p_adj_fmt, signif,
         eff, mean_a, mean_b, n_a, n_b) %>%
  pivot_wider(
    names_from  = contrast,
    values_from = c(p, p_adj, signif, eff, mean_a, mean_b, n_a, n_b),
    names_sep   = " | "
  ) %>%
  arrange(as.numeric(as.character(Time_Binned)))

# Nicely formatted gt table
p_table <- p_wide %>%
  gt(rowname_col = "Time_Binned") %>%
  tab_header(
    title = md("**Per-bin Statistical Comparison of Nuclear Area**"),
    subtitle = paste("Method:", test_method, "| p adjusted by BH (FDR)")
  ) %>%
  fmt_number(columns = where(is.numeric), decimals = 1) %>%
  tab_options(table.font.size = px(14))

p_table
