setwd("~/University of Wyoming/R Projects")

Membrane <- read.csv("Membrane Results.csv")

head(Membrane)

df <- data.frame(
  X = 1:6,
  Area = c(2390, 2218, 5766, 7508, 10211, 11841),
  Mean = c(202.132, 302.883, 311.711, 207.644, 172.827, 179.788),
  Min = c(139, 148, 127, 101, 108, 100),
  Max = c(321, 1070, 2721, 505, 440, 793),
  Circ. = c(0.209, 0.198, 0.807, 0.879, 0.905, 0.889),
  IntDen = c(483096, 671794, 1797323, 1558990, 1764737, 2128875),
  RawIntDen = c(483096, 671794, 1797323, 1558990, 1764737, 2128875),
  AR = c(1.628, 1.016, 1.342, 1.138, 1.055, 1.037),
  Round = c(0.614, 0.984, 0.745, 0.878, 0.948, 0.965),
  Solidity = c(0.884, 0.902, 0.972, 0.981, 0.985, 0.985)
)

# Insert a column for time
df$time <- as.POSIXct("00:00:00", format = "%H:%M:%S")

# Print the modified data frame
print(df)
