# import libraries and data ####################################################

library(tidyverse)
library(sf)
library(ggmap)
library(ggspatial)
source("code/helpers.R")

df <- read_csv("data/ACLED_Ukraine_2013-11-01-2024-12-16.csv")

# create buffers ###############################################################


# Convert to sf object
sf_points <- st_as_sf(df, coords = c("longitude", "latitude"), crs = 4326)

# Transform to a projected CRS for accurate distance calculations
sf_points_proj <- st_transform(sf_points, crs = 32633) # Example UTM zone

# Create buffers (e.g., 1000 meters)
buffers <- st_buffer(sf_points_proj, dist = 3500)

# Transform back to original CRS if needed
buffers <- st_transform(buffers, crs = 4326)

# plot them #####

## create admin2 bounding box ####
# find admin2 of Kyiv
admin2_val <- "Polohivskyi"

average_location <- df %>%
  filter(admin2 == admin2_val) %>%
  select(longitude, latitude) %>% 
  unique() %>% 
  summarise(longitude=mean(longitude),latitude=mean(latitude))

# Convert to sf object and transform CRS if necessary
admin2_sf <- st_as_sf(average_location, coords = c("longitude", "latitude"), crs = 4326)

# # Transform to a projected CRS for buffering
# admin2_sf_proj <- st_transform(admin2_sf, crs = 3857) # Using Web Mercator for example

admin2_buffer <- st_buffer(admin2_sf,
                         dist = 40000,
                         endCapStyle = "SQUARE")

# # Transform back to WGS 84 for plotting
# admin2_buffer <- st_transform(admin2_buffer, crs = 4326)

# create bbox from kiev buffer
admin2_bbox <- st_bbox(admin2_buffer)
# rename bbox columns
names(admin2_bbox) <- c("left", "bottom", "right", "top")

# get basemap
basemap <- get_stadiamap(
  bbox = admin2_bbox,
  zoom = 11,
  map_type = "toner-lite"
)

## create municipality buffer ####
# extract coordinates of all locations in admin2
municip <- df %>%
  filter(admin2 == admin2_val,
         geo_precision ==1) %>%
  select(longitude, latitude) %>%
  unique()

# Convert to sf object and transform CRS if necessary
municip_sf <- st_as_sf(municip, coords = c("longitude", "latitude"), crs = 4326)

# I dont think I need to transform it, because 4326 seems to work fine

# # Transform to a projected CRS for buffering
# municip_sf_proj <- st_transform(municip_sf, crs = 4326) # Using Web Mercator for example

municip_buffer <- st_buffer(municip_sf,
                           dist = 2500,
                           joinStyle = "ROUND")

# # Transform back to WGS 84 for plotting
# municip_buffer <- st_transform(municip_buffer, crs = 4326)


# plot
ggmap(basemap) +
  geom_sf(data = municip_buffer,
          fill = "blue",
          color = "blue",
          alpha = 0.2,
          inherit.aes = FALSE) +
  theme_void()
