####Bulk Analysis Test Quantification#### 
require(ggplot2) 
require(tidyverse)
library(scales)

####Read In Fiji Data To R#### 

#make sure you copy your save path where you have the fiji data
Nuclei_Data <- read.csv("C:/Users/tdeibert/Documents/UWYO/Nuclear Scaling Project/Code Repository/nuclear-scaling/Nuclear_Assembly_Test.csv")
library(tidyverse)

#Run this line to make your control data set. 
control_df <- Nuclei_Data %>%
  filter(str_detect(Sample.Name, "Control")) %>%
  filter(Time %in% c(15, 30, 45, 60)) %>% 
  arrange(Time, desc(Area)) %>% 
  mutate(Area = as.numeric(Area))

#Run this line to make the CDR2 data set
CDR2 <- Nuclei_Data %>% 
  filter(str_detect(Sample.Name, "Experiment")) %>%
  filter(Time %in% c(15, 30, 45, 60)) %>% 
  arrange(Time, desc(Area)) %>% 
  mutate(Area = as.numeric(Area))


####Make boxplot of Area at each timepoint####

# Control box plot for each time point with area. 
ggplot(control_df, aes(x = factor(Time), y = Area)) +
  geom_boxplot(outlier.shape = NA, fill = "grey", color = "black", alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.6, color = "black") +
  labs(
    title = "Area of Control Samples at Each Timepoint",
    x = "Time (minutes)",
    y = "Area"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +  # <- formats to 1 decimal place
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5)
  )
#CDR2 box plot for each time point with area. 
ggplot(CDR2, aes(x = factor(Time), y = Area)) +
  geom_boxplot(outlier.shape = NA, fill = "grey", color = "black", alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.6, color = "black") +
  labs(
    title = "Area of CDR2 Samples at Each Timepoint",
    x = "Time (minutes)",
    y = "Area"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +  # <- formats to 1 decimal place
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5)
  )










