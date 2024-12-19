# libraries and data #####

library(tidyverse)
library(sf)
library(ggmap)
library(ggspatial)
source("code/helpers.R")

df <- read_csv("data/ACLED_Ukraine_2013-11-01-2024-12-16.csv")

# variables
names(df)

# actors
show_top(df, actor1, 30)
show_top(df, actor2, 30)

# geo precision
show_top(df, geo_precision)

# event type
show_top(df, event_type)

# sub event type
show_top(df, sub_event_type)

# plot the frequency of number of fatalities
df %>%
  ggplot(aes(fatalities)) +
  scale_y_log10() +
  geom_histogram(binwidth = 1) +
  labs(title = "Frequency of number of fatalities",
       x = "Number of fatalities",
       y = "Frequency") +
  coord_cartesian(xlim = c(0, 200))



# plot on map

#### need a stadia or google api key for this 
register_stadiamaps("1b78d49b-bcdd-40ee-9d23-de210af45b6b", write = TRUE)

# Convert to a spatial data frame using sf
my_sf_data <- st_as_sf(df, coords = c("longitude", "latitude"), crs = 4326)

# Get a base map using ggmap
# Define the area of interest based on your data

bbox <- round(st_bbox(my_sf_data),2)

names(bbox) <- c("left", "bottom", "right", "top")


bbox <- c(left = 24.61, bottom = 59.37,
          right = 24.94, top = 59.5)


base_map <- get_stadiamap(bbox, zoom = 6, maptype = "stamen_terrain")
ggmap(base_map)

# filter data to only include rows where year is >= 2022
df_war <- df %>%
  filter(year >= 2022)

# Parameters for logarithmic breaks
min_density <- 0.0001
max_density <- 0.8
num_breaks <- 12
power_transform <- 0.5

# Generate logarithmic breaks with power transformation
log_breaks <- exp(log(min_density) + (seq(0, 1, length.out = num_breaks)^power_transform) * (log(max_density) - log(min_density)))



# Plot the heatmap
ggmap(base_map) +
  geom_point(data = df, aes(x = longitude, y = latitude),
             color = "red", alpha = 0.05, size = 0.1) +
  geom_density_2d_filled(data = df, aes(x = longitude, y = latitude,
                                            fill = ..level..),
                         alpha = 0.5,
                         # adjust=2,
                         breaks = log_breaks) +
  labs(title = "Heatmap on Geographic Map") +
  theme_minimal()

# save the map
ggsave("figures/heatmap_all_events.png", width = 10, height = 10, dpi = 300)
