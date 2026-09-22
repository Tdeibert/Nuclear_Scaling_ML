#learning to read and analyze raster data. 
library(stars)
library(ggplot2)
#lets get an raster image loaded into R. 
m_cherry_nls <- read_stars("S:/University Of Wyoming/Gatlin Lab/Nuclear Scaling Project/NE ecperiments/Control Conditions/Control/Extract 3/Region 1/Droplet 1/Mcherry-10.tif")


ggplot() +
  geom_stars(data=m_cherry_nls)


