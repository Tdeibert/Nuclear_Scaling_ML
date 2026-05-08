T1 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T1_NPC.csv")
T2 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T2_NPC.csv")
T3 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T3_NPC.csv")
T4 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T4_NPC.csv")
T5 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T5_NPC.csv")
T6 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T6_NPC.csv")
T7 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T7_NPC.csv")
T8 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T8_NPC.csv")
T9 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T9_NPC.csv")
T10 <- read.csv("D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/T10_NPC.csv")
Time_Point_1_10 <- bind_rows(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)
write_csv(Time_Point_1_10, "D:/Membrane Experiments/P150/P150 Cc1/Outputs_Non_Filtered/Time_1_10.csv")
Full_Data_Set <- Time_Point_1_10 %>% 
  select(cluster, N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area, Slice, X, Y, Region) %>%  
  rename(Time = Region)


#Full_Data_Set <- Full_Data_Set %>% drop_na(N_C_V2, Time)

Full_Data_Set <- Full_Data_Set %>%
  drop_na(N_C_V2, Time) %>% 
  filter(!apply(. < 0, 1, any)) %>% 
  filter(!(Time > 30 & N_C_V2 < 0.7)) %>% 
  filter(!(Time > 20 & Nuclear_Area < 100)) %>% 
  filter(!(Time > 20 & Nuclear_Area > 400)) %>% 
  filter(!(Nuclear_Membrane_BG_Subtracted > 30000)) %>% 
  filter(!(Nuclear_Membrane_BG_Subtracted < 5)) %>% 
  filter(!(Nuclear_Area < 20))



Full_Data_Set <- Full_Data_Set %>%
  arrange(desc(Time))


library(tidyverse)

# Step 1: Create 6-minute time clusters
Full_Data_Set_Clustered <- Full_Data_Set %>%
  mutate(Time_Cluster = floor(Time / 6) * 6)

# Step 2: Create the boxplot
ggplot(Full_Data_Set_Clustered, aes(x = factor(Time_Cluster), y = Nuclear_Area)) +
  geom_boxplot(fill = "grey", color = "black", outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.2, size = 1.5, alpha = 0.6, color = "black") +
  labs(
    title = "P150 Nuclear Area by 6-Minute Time Clusters",
    x = "Time Cluster (minutes)",
    y = "Nuclear Area"
  ) +
  theme_minimal()



Control_Full_Data <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Data Sets/Control/Extract 3/Bulk Image Analysis All Time Points/Outputs_3/Time_Point_1_10.csv") %>% 
  select(cluster, N_C_V2, Nuclear_Membrane_BG_Subtracted, Nuclear_Area, Slice, X, Y, Region) %>%  
  rename(Time = Region)

Control_Full_Data <- Control_Full_Data %>%
  filter(!apply(. < 0, 1, any)) %>% 
  filter(!(Time > 30 & N_C_V2 < 0.7)) %>% 
  filter(!(Time > 40 & Nuclear_Area < 250)) %>% 
  filter(!(Nuclear_Membrane_BG_Subtracted > 30000)) %>% 
  filter(!(Nuclear_Membrane_BG_Subtracted < 5000)) %>% 
  filter(!(Nuclear_Area < 20))

Control_Full_Data_Set_Clustered <- Control_Full_Data %>%
  mutate(Time_Cluster = floor(Time / 6) * 6)

ggplot(Control_Full_Data_Set_Clustered, aes(x = factor(Time_Cluster), y = Nuclear_Area)) +
  geom_boxplot(fill = "grey", color = "black", outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.2, size = 1.5, alpha = 0.6, color = "black") +
  labs(
    title = "Control Area by 6-Minute Time Clusters",
    x = "Time Cluster (minutes)",
    y = "Nuclear Area"
  ) +
  theme_minimal()


#### Statistic ####  Add a condition column to each dataset
P150 <- Full_Data_Set %>% mutate(Condition = "P150")
Control <- Control_Full_Data %>% mutate(Condition = "Control")

# Bind them together
Combined_Data <- bind_rows(P150, Control)

# Make sure time clusters are created
Combined_Data <- Combined_Data %>%
  mutate(Time_Cluster = floor(Time / 6) * 6)

anova_result <- aov(Nuclear_Area ~ Time_Cluster * Condition, data = Combined_Data)
summary(anova_result)

# Split data by Time_Cluster and compare
Combined_Data %>%
  group_by(Time_Cluster) %>%
  summarise(
    p_value = t.test(Nuclear_Area ~ Condition)$p.value
  )

library(ggpubr)

ggplot(Combined_Data, aes(x = factor(Time_Cluster), y = Nuclear_Area, fill = Condition)) +
  geom_boxplot(position = position_dodge(0.9)) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.9), alpha = 0.5) +
  stat_compare_means(aes(group = Condition), method = "t.test", label = "p.signif", 
                     label.y = 400, size = 3) +
  labs(title = "Nuclear Area Comparison", x = "Time Cluster (min)", y = "Nuclear Area") +
  theme_minimal()
