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

# calculate pairwise distance between locations within same admin1 #############

library(furrr)
library(tidyverse)

plan(multisession, workers = 4)

# create df with unique locations
unique_locations <- df %>%
  filter(geo_precision ==1,
         event_type=="Explosions/Remote violence") %>%
  select(location, admin1, longitude, latitude) %>%
  unique()

# Convert to sf object and transform CRS if necessary
sf_df <- st_as_sf(unique_locations, coords = c("longitude", "latitude"), crs = 4326)

# Create a function to calculate the distance between two points
calculate_distance <- function(point1, point2) {
  return(st_distance(point1, point2))
}

polygon_pairs <- expand.grid(1:nrow(sf_buffer), 1:nrow(sf_buffer))

# Remove pairs where the indices are the same (i.e., the same polygon)
polygon_pairs <- polygon_pairs[polygon_pairs$Var1 != polygon_pairs$Var2, ]

# remove polygon pairs that are not in the same admin1
polygon_pairs <- polygon_pairs %>%
  mutate(admin1_1 = sf_df$admin1[Var1],
         admin1_2 = sf_df$admin1[Var2]) %>%
  filter(admin1_1 == admin1_2)

# distance results
distance_results <- polygon_pairs %>%
  mutate(distance = future_pmap_dbl(
    list(Var1, Var2),
    ~ calculate_distance(sf_df$geometry[.x], sf_df$geometry[.y])
  ))

# add location1, location2, geometry1, and geometry 2 to distance_results
distance_results2 <- distance_results %>%
  mutate(location1 = sf_df$location[Var1],
         location2 = sf_df$location[Var2],
         geometry1 = sf_df$geometry[Var1],
         geometry2 = sf_df$geometry[Var2])

# save results
write_csv(distance_results2, "data/distance_between_locations.csv")

# calculate pairwise overlap between buffers ##################################

library(progress)

pb <- progress_bar$new(
  format = "  Processing [:bar] :percent in :elapsed",
  total = nrow(polygon_pairs),
  clear = FALSE,
  width = 60
)

# create df with unique locations
unique_locations <- df %>%
  filter(geo_precision ==1,
         event_type=="Explosions/Remote violence") %>%
  select(location, admin1, longitude, latitude) %>%
  unique()

# Convert to sf object and transform CRS if necessary
sf_df <- st_as_sf(unique_locations, coords = c("longitude", "latitude"), crs = 4326)

# I dont think I need to transform it, because 4326 seems to work fine

# # Transform to a projected CRS for buffering
# municip_sf_proj <- st_transform(municip_sf, crs = 4326) # Using Web Mercator for example

sf_buffer <- st_buffer(sf_df,
                            dist = 2500,
                            joinStyle = "ROUND")


# Create a function to calculate the intersection area between two polygons
calculate_overlap <- function(poly1, poly2) {
  intersection <- st_intersection(poly1, poly2)
  if (length(intersection) > 0) {
    return(st_area(intersection[2]))
  } else {
    return(0)
  }
}

# Get all combinations of polygon pairs
polygon_pairs <- expand.grid(1:nrow(sf_buffer), 1:nrow(sf_buffer))

# Remove pairs where the indices are the same (i.e., the same polygon)
polygon_pairs <- polygon_pairs[polygon_pairs$Var1 != polygon_pairs$Var2, ]

# remove polygon pairs that are not in the same admin1
polygon_pairs <- polygon_pairs %>%
  mutate(admin1_1 = sf_df$admin1[Var1],
         admin1_2 = sf_df$admin1[Var2]) %>%
  filter(admin1_1 == admin1_2)

# Calculate overlap for each pair
overlap_results <- polygon_pairs %>%
  rowwise() %>%
  mutate(overlap_area = {
    # Update the progress bar
    pb$tick()
    
    # Calculate the overlap area
    calculate_overlap(sf_buffer$geometry[Var1], sf_buffer$geometry[Var2])
  }) %>%
  ungroup()

# Create a new dataframe with the results
overlap_df <- overlap_results %>%
  select(polygon1 = Var1, polygon2 = Var2, overlap_area)

# View the results
print(overlap_df)
