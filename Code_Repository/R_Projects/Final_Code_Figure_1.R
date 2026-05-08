library(tidyverse)
library(ggplot2)
library(readxl)
library(ggpubr)
library(rstatix)
library(gt)
library(paletteer)


#### Color Codes#### 
Control = "#117733"
P150    = "#88CCEE"
CDR2    = "#882255"
N_C     = "black"



#### Control Figures#### 
Control_Data_1_2 <- read.csv("C:/Users/tdeibert/.Working_Docs_Folder/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_3/Time_Point_1_10.csv")


Control_Data_1_2 <- Control_Data_1_2 %>% 
  mutate(Time = Region)

Control_Binned <- Control_Data_1_2 %>%  
  mutate(Time_Binned = floor(Time/6)* 6)



#Single Axis Control Growth Plot 

ggplot( data = Control_Data_1_2, aes(x = Time, y = Nuclear_Area,))+
  geom_point(size = 2, color = "black", fill = Control, alpha = 0.8, shape = 21)+
  geom_smooth (method = "loess", color = Control) +
  scale_x_continuous(limits = c(0,90), breaks = c(0,10, 20, 30, 40, 50, 60, 70, 80, 90))+
  scale_y_continuous(limits = c(0,500), breaks = c(0,100,200,300,400,500)) +
  labs (
    y = "nuclear cross-sectional Area", 
    x = "time (minutes)" )



#Single Axis Control Data N/C plot 

ggplot(data = Control_Data_1_2, aes(x = Time, y = N_C_V2))+
  geom_point(size = 2, color = "black", fill = N_C, alpha = 0.5, shape = 21)+
  geom_smooth(method = "loess", color = "black") +
  scale_x_continuous(limits = c(0,90), breaks = c(0,10,20,30,40,50,60,70,80,90))+
  scale_y_continuous(limits = c(0,1), breaks = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))+
  labs(
    y = "N/C Ratio",
    x = "Time"
  )

#Single Axis Control Box Plots 

ggplot(data = Control_Binned, aes(x = Time_Binned, y = Nuclear_Area, group = Time_Binned)) +
  geom_boxplot(
    width = 3,              # wider boxes to match 6-min intervals visually
    alpha = 0.7,
    color = "black",
    fill = Control,
    shape = 1,
    outlier.shape = NA
  ) +
  geom_jitter(width = 1.5, alpha = 0.3, color = Control, size = 1) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 6),       # ticks every 6 minutes
    name = "time (minutes)"
  ) +
  scale_y_continuous(
    limits = c(0, 500),
    breaks = seq(0, 500, by = 100),
    name = "nuclear cross-sectional area (um^2)"
  ) +
  theme_classic(base_size = 14)


# Duel Axis Plot 


# Define scaling factor to map N/C ratio (0–1) onto Nuclear_Area range (0–600)
scale_factor <- 500  # 1 on N/C axis = 600 on Nuclear_Area axis


ggplot(Control_Data_1_2, aes(x = Time)) +
  # Plot N/C Ratio (left axis)
  geom_point(aes(y = N_C_V2), size = 2, color = "black", fill = N_C, alpha = 0.5, shape = 21) +
  geom_smooth(aes(y = N_C_V2), method = "loess", color = N_C) +
  
  # Plot Nuclear Area (right axis, scaled)
  geom_point(aes(y = Nuclear_Area / scale_factor), size = 2, color = "black", fill = Control, alpha = 0.5, shape = 21) +
  geom_smooth(aes(y = Nuclear_Area / scale_factor), method = "loess", color = Control) +
  
  # X axis
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, 10)) +
  
  # Primary Y (N/C Ratio) + Secondary Y (Nuclear Area)
  scale_y_continuous(
    name = "n/c ratio",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1),
    sec.axis = sec_axis(~ . * scale_factor, name = "nuclear cross-sectional area(um^2)")
  ) +
  labs(
    x = "time (min)",
    y = "n/c ratio"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    axis.title.y.left  = element_text(color = N_C),
    axis.text.y.left   = element_text(color = N_C),
    axis.title.y.right = element_text(color = "black"),
    axis.text.y.right  = element_text(color = "black")
  )


#-------------------------------------------------------------------------------------------------------------------


####P150#### 
#P150_7_17 <-read.csv("D:/Membrane Experiments/P150/7-17/Full_Data_Set_Processed.csv") 
#P150_4_23 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs/Full_Data_Set.csv")

#P150_Data <- bind_rows(
#  P150_4_23 %>% mutate(X.1 = parse_number(as.character(X.1))),
#  P150_7_17 %>% mutate(X.1 = parse_number(as.character(X.1)))
#)

#write.csv(P150_Data, "C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/P150/p150_combined.csv")

P150_Data <- read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/P150/p150_combined.csv")
P150_filtered <- P150_Data %>%
  # Step 1: Remove small nuclei after 30 minutes
  filter(Time < 15 | Nuclear_Area >= 50) %>%
  
  # Step 2: Remove points more than 2 SD from the mean Nuclear_Area
  filter(
    Nuclear_Area >= mean(Nuclear_Area, na.rm = TRUE) - 2 * sd(Nuclear_Area, na.rm = TRUE),
    Nuclear_Area <= mean(Nuclear_Area, na.rm = TRUE) + 2 * sd(Nuclear_Area, na.rm = TRUE)
  )

P150_Binned <- P150_filtered %>%  
  mutate(Time_Binned = floor(Time/6)* 6)

#Single Axis Control Growth Plot 
ggplot( data = P150_filtered, aes(x = Time, y = Nuclear_Area,))+
  geom_point(size = 2, color = "black", fill = P150, alpha = 0.8, shape = 21)+
  geom_smooth (method = "loess", color = "black") +
  scale_x_continuous(limits = c(0,90), breaks = c(0,10, 20, 30, 40, 50, 60, 70, 80, 90))+
  scale_y_continuous(limits = c(0,500), breaks = c(0,100,200,300,400,500)) +
  labs (
    y = "Nuclear Cross Sectional Area", 
    x = "Time" )

#Single Axis P150 Data N/C plot 
ggplot(data = P150_filtered, aes(x = Time, y = N_C_V2))+
  geom_point(size = 2, color = "black", fill = N_C, alpha = 0.5, shape = 21)+
  geom_smooth(method = "loess", color = "black") +
  scale_x_continuous(limits = c(0,90), breaks = c(0,10,20,30,40,50,60,70,80,90))+
  scale_y_continuous(limits = c(0,1), breaks = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))+
  labs(
    y = "N/C Ratio",
    x = "Time"
  )

#Duel Axis Plot

#Define scaling factor to map N/C ratio (0–1) onto Nuclear_Area range (0–600)
scale_factor <- 500  # 1 on N/C axis = 600 on Nuclear_Area axis

ggplot(P150_filtered, aes(x = Time)) +
  # Plot N/C Ratio (left axis)
  geom_point(aes(y = N_C_V2), size = 2, color = "black", fill = N_C, alpha = 0.5, shape = 21) +
  geom_smooth(aes(y = N_C_V2), method = "loess", color = "black") +
  
  # Plot Nuclear Area (right axis, scaled)
  geom_point(aes(y = Nuclear_Area / scale_factor), size = 2, color = "black", fill = P150, alpha = 0.8, shape = 21) +
  geom_smooth(aes(y = Nuclear_Area / scale_factor), method = "loess", color = P150) +
  
  # X axis
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, 10)) +
  
  # Primary Y (N/C Ratio) + Secondary Y (Nuclear Area)
  scale_y_continuous(
    name = "n/c ratio",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1),
    sec.axis = sec_axis(~ . * scale_factor, name = "nuclear cross-sectional area (um^2)")
  ) +
  
  labs( x = "time (minutes)") +
  theme_classic(base_size = 14) +
  theme(
    axis.title.y.left = element_text(color = N_C),
    axis.title.y.right = element_text(color = "black"),
    axis.text.y.right = element_text(color = "black"),
    axis.text.y.left = element_text(color = N_C)
  )

#Single Axis P150 Box Plots 
ggplot(data = P150_Binned, aes(x = Time_Binned, y = Nuclear_Area, group = Time_Binned)) +
  geom_boxplot(
    width = 3,              # wider boxes to match 6-min intervals visually
    alpha = 0.8,
    color = "black",
    fill = P150,
    shape = 1,
    outlier.shape = NA
  ) +
  geom_jitter(width = 1.5, alpha = 0.3, color = P150, size = 1) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 6),       # ticks every 6 minutes
    name = "time (minutes)"
  ) +
  scale_y_continuous(
    limits = c(0, 500),
    breaks = seq(0, 500, by = 100),
    name = "nuclear cross-sectional area (um^2)"
  ) +
  theme_classic(base_size = 14)

#### CDR2 ####
#CDR2_6_25 <- read.csv("D:/CDR2 for Analysis/CDR2/full_data_set.csv")
#CDR2_9_27 <- read.csv("D:/CDR2 for Analysis/CDR2_25_09_07/Incubation_Full_100_Min.csv")

CDR2_6_25 <- read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/CDR2/Data_Sets/full_data_set_5_29.csv")
CDR2_9_27 <- read.csv("C:/Users/cowboy/OneDrive/Documents/Unviversity of Alabama/Nuclear_Scaling/Data_Files/CDR2/Data_Sets/Incubation_Full_100_Min_02_25_2025.csv")


CDR2_Data <- bind_rows(CDR2_6_25,CDR2_9_27)
CDR2_Binned <- CDR2_Data %>%  
  mutate(Time_Binned = floor(Time/6)* 6)


# CDR2 Area Single Axis Plot 
ggplot( data = CDR2_Data, aes(x = Time, y = Nuclear_Area,))+
  geom_point(size = 2, color = "black", fill = CDR2, alpha = .5, shape = 21)+
  geom_smooth (method = "loess", color = "black") +
  scale_x_continuous(limits = c(0,90), breaks = c(0,10, 20, 30, 40, 50, 60, 70, 80, 90))+
  scale_y_continuous(limits = c(0,500), breaks = c(0,100,200,300,400,500)) +
  labs (
    y = "Nuclear Cross Sectional Area", 
    x = "Time" )

#Single Axis Control Data N/C plot 
ggplot(data = CDR2_Data, aes(x = Time, y = N_C_V2))+
  geom_point(size = 2, color = "black", fill = N_C, alpha = 0.5, shape = 21)+
  geom_smooth(method = "loess", color = "black") +
  scale_x_continuous(limits = c(0,90), breaks = c(0,10,20,30,40,50,60,70,80,90))+
  scale_y_continuous(limits = c(0,1), breaks = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))+
  labs(
    y = "N/C Ratio",
    x = "Time"
  )

# Define scaling factor to map N/C ratio (0–1) onto Nuclear_Area range (0–600)
scale_factor <- 500  # 1 on N/C axis = 600 on Nuclear_Area axis

ggplot(CDR2_Data, aes(x = Time)) +
  # Plot N/C Ratio (left axis)
  geom_point(aes(y = N_C_V2), size = 2, color = "black", fill = N_C, alpha = 0.5, shape = 21) +
  geom_smooth(aes(y = N_C_V2), method = "loess", color = "black") +
  
  # Plot Nuclear Area (right axis, scaled)
  geom_point(aes(y = Nuclear_Area / scale_factor), size = 2, color = "black", fill = CDR2, alpha = .8, shape = 21) +
  geom_smooth(aes(y = Nuclear_Area / scale_factor), method = "loess", color = CDR2) +
  
  # X axis
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 90, 10)) +
  
  # Primary Y (N/C Ratio) + Secondary Y (Nuclear Area)
  scale_y_continuous(
    name = "n/c ratio",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1),
    sec.axis = sec_axis(~ . * scale_factor, name = "nuclear cross-sectional area um^2")
  ) +
  
  labs(x = "time (minutes)") +
  theme_classic(base_size = 14) +
  theme(
    axis.title.y.left = element_text(color = N_C),
    axis.title.y.right = element_text(color = "black"),
    axis.text.y.right = element_text(color = "black"),
    axis.text.y.left = element_text(color = N_C)
  )

ggplot(data = CDR2_Binned, aes(x = Time_Binned, y = Nuclear_Area, group = Time_Binned)) +
  geom_boxplot(
    width = 3,              # wider boxes to match 6-min intervals visually
    alpha = 1,
    color = "black",
    fill = CDR2,
    shape = 1,
    outlier.shape = NA
  ) +
  geom_jitter(width = 1.5, alpha = 0.3, color = CDR2, size = 1) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 6),       # ticks every 6 minutes
    name = "time (minutes)"
  ) +
  scale_y_continuous(
    limits = c(0, 500),
    breaks = seq(0, 500, by = 100),
    name = "nuclear cross-sectional area (um^2)"
  ) +
  theme_classic(base_size = 14)


#### Comparing Data #### 
# 1) Combine all datasets
Control_Binned <- Control_Binned %>% mutate(Condition = "Control")
P150_Binned    <- P150_Binned    %>% mutate(Condition = "P150")
CDR2_Binned    <- CDR2_Binned    %>% mutate(Condition = "CDR2")

Combined_Binned <- bind_rows(
  Control_Binned %>% select(-any_of("X.1")),
  P150_Binned    %>% select(-any_of("X.1")),
  CDR2_Binned    %>% select(-any_of("X.1"))
) %>%
  mutate(
    Time_Binned = factor(Time_Binned, levels = sort(unique(Time_Binned))),
    Condition   = factor(Condition, levels = c("Control","P150","CDR2"))
  )



# 2) Function to compute per-bin pairwise tests
pairwise_bin_test <- function(dat, cond_a, cond_b, method = "t.test") {
  dat %>%
    filter(Condition %in% c(cond_a, cond_b)) %>%
    group_by(Time_Binned) %>%
    summarise(
      p = {
        x <- Nuclear_Area[Condition == cond_a]
        y <- Nuclear_Area[Condition == cond_b]
        if (length(x) > 1 && length(y) > 1) {
          if (method == "t.test") t.test(x, y)$p.value else wilcox.test(x, y)$p.value
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

# 3) Build both comparisons
p_P150 <- pairwise_bin_test(Combined_Binned, "Control", "P150", method = "t.test")
p_CDR2 <- pairwise_bin_test(Combined_Binned, "Control", "CDR2", method = "t.test")

# 4) Get y positions for p-value labels
ypos <- Combined_Binned %>%
  group_by(Time_Binned) %>%
  summarise(ymax = max(Nuclear_Area, na.rm = TRUE), .groups = "drop")

p_all <- bind_rows(p_P150, p_CDR2) %>%
  left_join(ypos, by = "Time_Binned") %>%
  arrange(Time_Binned, group2) %>%
  group_by(Time_Binned) %>%
  mutate(y.position = ymax + 10 + 10 * row_number()) %>%
  ungroup()


# Make a clean vector of the actual x levels used in the plot
bin_levels <- levels(Combined_Binned$Time_Binned) %||% as.character(sort(unique(Combined_Binned$Time_Binned)))

p_all <- bind_rows(p_P150, p_CDR2) %>%
  dplyr::filter(!is.na(Time_Binned)) %>%
  dplyr::left_join(
    Combined_Binned %>%
      dplyr::group_by(Time_Binned) %>%
      dplyr::summarise(ymax = max(Nuclear_Area, na.rm = TRUE), .groups = "drop"),
    by = "Time_Binned"
  ) %>%
  dplyr::mutate(
    Time_Binned = factor(Time_Binned, levels = bin_levels),
    
    # base position just above the tallest box
    y.position  = ymax + 12,
    
    # now P150 brackets are higher; CDR2 stays at the base
    y.position  = ifelse(group2 == "P150", y.position + 22, y.position),
    
    # left side of bracket is always Control
    group1 = "Control"
  ) %>%
  dplyr::mutate(
    p_label          = sprintf("p = %.2g", p),
    y.position.label = y.position + 15
  )



# data cleaning for brackets 
p_brackets <- p_all %>%
  filter(p.signif != "ns") %>%   # remove NS
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
    
    # color code by “other” group (group2)
    star_color = case_when(
      group2 == "P150" ~ P150,   # teal
      group2 == "CDR2" ~ CDR2   # gold
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



####Significance Plots#### 

Combined_Binned <- Combined_Binned %>%
  dplyr::mutate(
    Condition = factor(Condition,
                       levels = c("Control", "P150", "CDR2"))
  )



#no barackets only stars 

early_bins <- c("6","12","18","24","30","36","42","48","54","60")
late_bins  <- c("66","72","78","84","90","96")

ggplot(Combined_Binned,
       aes(x = Time_Binned, y = Nuclear_Area, fill = Condition)) +
  geom_boxplot(
    position = position_dodge2(width = 0.75, preserve = "single"),
    width    = 0.65,
    alpha    = 0.85,
    color    = "black",
    outlier.shape = NA
  ) +
  ggpubr::stat_pvalue_manual(
    p_all,
    label        = "p.signif",
    tip.length   = 0.01,
    bracket.size = 0.4,
    step.increase= 0,
    x            = "Time_Binned",
    xmin         = "group1",
    xmax         = "group2",
    y.position   = "y.position",
    hide.ns      = TRUE
  ) +
  scale_y_continuous(limits = c(0, 600), breaks = seq(0, 600, 100)) +
  scale_fill_manual(
    values = c(
      "Control" = Control,
      "P150"    = P150,
      "CDR2"    = CDR2
    ),
    breaks = c("Control", "P150", "CDR2"),   # legend order
    labels = c("Control", "p150-CC1", "CDR2"),
    name   = NULL                            # remove legend title
  ) +
  labs(
    x    = "time binned (min)",
    y    = "nuclear cross-sectional area (µm^2)",
    fill = NULL                              # also makes sure no title
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  ) +
  coord_cartesian(clip = "off")

#########

ggplot(Combined_Binned,
       aes(x = Time_Binned, y = Nuclear_Area, fill = Condition)) +
  geom_boxplot(
    position = position_dodge2(width = 0.75, preserve = "single"),
    width    = 0.65,
    alpha    = 0.85,
    color    = "black",
    outlier.shape = NA
  ) +
  
  # Horizontal bracket line
  geom_segment(
    data = p_brackets,
    aes(x = x1, xend = x2, y = y.position, yend = y.position),
    inherit.aes = FALSE,
    linewidth   = 0.6
  ) +
  # Left tick
  geom_segment(
    data = p_brackets,
    aes(x = x1, xend = x1, y = y.position, yend = y.position - 5),
    inherit.aes = FALSE,
    linewidth   = 0.6
  ) +
  # Right tick
  geom_segment(
    data = p_brackets,
    aes(x = x2, xend = x2, y = y.position, yend = y.position - 5),
    inherit.aes = FALSE,
    linewidth   = 0.6
  ) +
  # Stars
  geom_text(
    data = p_brackets,
    aes(x = xmid, y = y.position + 5, label = p.signif, color = star_color),
    inherit.aes = FALSE,
    size        = 5
  ) +
  
  # Prevent star_color from creating a legend
  scale_color_identity(guide = "none") +
  
  scale_y_continuous(limits = c(0, 600), breaks = seq(0, 600, 100)) +
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

# 3) Early-only stats (start from your existing p_all)
p_all_early <- p_all %>%
  filter(Time_Binned %in% early_bins) %>%
  mutate(Time_Binned = droplevels(Time_Binned))

# 4) Rebuild bracket table (no "ns", color-coded)
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
    
    # color by the non-control group
    star_color = case_when(
      group2 == "P150" ~ P150,
      group2 == "CDR2" ~ Control
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

# 5) Plot: early time points only
ggplot(Combined_early,
       aes(x = Time_Binned, y = Nuclear_Area, fill = Condition)) +
  geom_boxplot(
    position = position_dodge2(width = 0.75, preserve = "single"),
    width    = 0.65,
    alpha    = 0.85,
    color    = "black",
    outlier.shape = NA
  ) +
  # brackets + stars (color-coordinated)
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
  scale_fill_manual(values = c(
    "Control" = Control,
    "P150"    =  P150,
    "CDR2"    = CDR2
  )) +
  scale_color_identity() +
  scale_y_continuous(limits = c(0, 600), breaks = seq(0, 600, 100)) +
  scale_fill_manual(
    values = c(
      "Control" = Control,
      "P150"    = P150,
      "CDR2"    = CDR2
    ),
    breaks = c("Control", "P150", "CDR2"),   # legend order
    labels = c("Control", "p150-CC1", "CDR2"),
    name   = NULL                            # remove legend title
  ) +
  labs(
    x    = "time binned (min)",
    y    = "nuclear cross-sectional area (µm^2)",
    fill = NULL                              # also makes sure no title
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  ) +
  coord_cartesian(clip = "off")


####Table#### 

test_method <- "t.test"  # change to "wilcox.test" if preferred
digits_p    <- 3         # display precision for p-values

# --- Helper: run a two-sample test and return p + summary -------------------
bin_test_once <- function(df, cond_a, cond_b, method = c("t.test","wilcox.test")[1]) {
  x <- df$Nuclear_Area[df$Condition == cond_a]
  y <- df$Nuclear_Area[df$Condition == cond_b]
  if (length(x) < 2 || length(y) < 2) {
    tibble(p = NA_real_, eff = NA_real_, mean_a = mean(x), mean_b = mean(y),
           n_a = length(x), n_b = length(y))
  } else {
    if (method == "t.test") {
      tt <- t.test(x, y)
      tibble(p = tt$p.value,
             eff = (mean(y) - mean(x)),        # simple mean difference (B - A)
             mean_a = mean(x), mean_b = mean(y),
             n_a = length(x), n_b = length(y))
    } else {
      ww <- wilcox.test(x, y, exact = FALSE)
      tibble(p = ww$p.value,
             eff = (median(y) - median(x)),    # median difference for Wilcox
             mean_a = mean(x), mean_b = mean(y),
             n_a = length(x), n_b = length(y))
    }
  }
}

# --- Build long table of tests for each bin/contrast ------------------------
# Assumes Combined_Binned exists with columns: Time_Binned, Condition, Nuclear_Area
contrasts <- tribble(
  ~cond_a,   ~cond_b,   ~contrast,
  "Control", "P150",    "P150 vs Control",
  "Control", "CDR2",    "CDR2 vs Control"
)

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
            ..1, ..2,
            method = test_method
          )
        )
      ) %>%
      tidyr::unnest(res)
  }) %>%
  ungroup() %>%
  # FDR within each contrast across time bins
  group_by(contrast) %>%
  mutate(p_adj_FDR = p.adjust(p, method = "BH")) %>%
  ungroup() %>%
  mutate(
    p_fmt     = ifelse(is.na(p), NA, formatC(p, format = "e", digits = digits_p)),
    p_adj_fmt = ifelse(is.na(p_adj_FDR), NA, formatC(p_adj_FDR, format = "e", digits = digits_p)),
    signif    = dplyr::case_when(
      is.na(p_adj_FDR) ~ "ns",
      p_adj_FDR < 1e-4 ~ "****",
      p_adj_FDR < 1e-3 ~ "***",
      p_adj_FDR < 1e-2 ~ "**",
      p_adj_FDR < 0.05 ~ "*",
      TRUE             ~ "ns"
    )
  )


# --- Wide view: one row per time bin, columns per contrast ------------------
p_wide <- p_long %>%
  select(Time_Binned, contrast, p = p_fmt, p_adj = p_adj_fmt, signif,
         eff, mean_a, mean_b, n_a, n_b) %>%
  pivot_wider(
    names_from = contrast,
    values_from = c(p, p_adj, signif, eff, mean_a, mean_b, n_a, n_b),
    names_sep = " | "
  ) %>%
  arrange(as.numeric(as.character(Time_Binned)))

# --- Pretty table with gt ---------------------------------------------------
p_table <- p_wide %>%
  gt(rowname_col = "Time_Binned") %>%
  tab_header(
    title = md("**Per-bin Statistical Comparison of Nuclear Area**"),
    subtitle = paste("Method:", test_method, " | p adjusted by BH (FDR)")
  ) %>%
  fmt_number(columns = where(is.numeric), decimals = 1) %>%
  cols_label(.list = setNames(names(p_wide)[-1], names(p_wide)[-1])) %>%
  tab_options(table.font.size = px(14)) %>%
  tab_spanner_delim(delim = " //| ") %>%   # groups columns by contrast automatically
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_spanners(everything())
  )

# View the table in RStudio Viewer
p_table
