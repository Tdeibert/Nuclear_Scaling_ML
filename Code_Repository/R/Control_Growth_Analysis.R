#Control addback vs actin intact quantification. 
require(tidyverse)
require(openxlsx2)

Actin_Intact<- read_xlsx("./Control_Growth.xlsx", sheet = "Actin_Intact_AddBack")
Actin_Free <- read_xlsx("./Control_Growth.xlsx", sheet = "Actin_Free")

require(tidyverse)

# Select the "Area" column from both data frames
df1_selected <- Actin_Intact %>% select(Area)
df2_selected <- Actin_Free %>% select(Area)

df1_selected <- df1_selected %>% 
  rename(Actin_Intact_Area = Area)

df2_selected <- df2_selected %>% 
  rename(Actin_Free_Area = Area)

# Merge into one data frame
merged_df <- bind_rows(df1_selected, df2_selected)

require(tidyverse)

# Convert to long format for ggplot
long_df <- merged_df %>%
  pivot_longer(cols = everything(), names_to = "Condition", values_to = "Area")

# Create the box plot
ggplot(long_df, aes(x = Condition, y = Area, fill = Condition)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +  # Makes boxplot slightly transparent & removes default outliers
  geom_jitter(width = 0.2, size = 2, alpha = 0.7, aes(color = Condition)) +  # Adds jittered data points
  theme_minimal() +
  labs(title = "Actin Free vs 100 Percent Addback",
       x = "120 Minutes",
       y = "Cross Sectional Area") +
  scale_fill_manual(values = c("grey", "grey")) +  # Box colors
  scale_color_manual(values = c("black", "black")) +  # Jitter point colors
  theme(legend.position = "none")  # Remove legend


# View the result
head(merged_df)
