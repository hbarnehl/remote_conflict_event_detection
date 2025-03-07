# import libraries and data ####################################################

library(tidyverse)
library(sf)
library(ggmap)
library(ggspatial)
source("code/helpers.R")

df <- read_csv("data/ACLED_Sudan_2021-01-01-2025-01-21.csv")

# calculate pairwise distance between locations within same admin1 #############

library(furrr)

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

# create all possible pairs of polygons
polygon_pairs <- expand.grid(1:nrow(sf_df), 1:nrow(sf_df))

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
write_csv(distance_results2, "data/sudan_distance_between_locations.csv")

# Create places data frame #####################################################
places <- df %>% 
  distinct(location, latitude, longitude, admin1, admin2, admin3) %>% 
  arrange(location) %>% 
  mutate(location_id = row_number())

# Convert to sf object and transform CRS if necessary
places_buffer <- st_as_sf(places, coords = c("longitude", "latitude"), crs = 4326)

places_buffer <- st_buffer(places_buffer,
                            dist = 3000,
                            joinStyle = "ROUND")

# change geometry vars into lon and lat vars
distance_df <- distance_results2 %>%
  mutate(geometry1 = str_remove_all(geometry1, "c\\(|\\)"),
         geometry2 = str_remove_all(geometry2, "c\\(|\\)")) %>%
  separate(geometry1, into = c("longitude1", "latitude1"), sep = ", ", convert = TRUE) %>% 
  separate(geometry2, into = c("longitude2", "latitude2"), sep = ", ", convert = TRUE)

distance_df_1 <- distance_df %>%
  left_join(places %>%
              select(location_id, location, longitude, latitude) %>% 
              rename(location_id_1=location_id),
            by = c("location1" = "location",
                   "longitude1" = "longitude",
                   "latitude1" = "latitude")) %>% 
  left_join(places %>%
              select(location_id, location, longitude, latitude) %>% 
              rename(location_id_2=location_id),
            by = c("location2" = "location",
                   "longitude2" = "longitude",
                   "latitude2" = "latitude"))

# remove admin, location, latitude, and longitude columns
distance_df_1 <- distance_df_1 %>%
  select(-c(admin1_1, admin1_2, location1, location2, longitude1, latitude1, longitude2, latitude2))

# save results
write_csv(distance_df_1, "data/sudan_distance_between_locations.csv")

# per location1, aggregate location ids of location2 into string where distance is less than 3000
loc_less_3000 <- distance_df_1 %>%
  filter(distance < 3000) %>%
  group_by(location_id_1) %>%
  summarise(location_id_2 = paste(location_id_2, collapse = ", "))

# join into places
places <- places_buffer %>%
  left_join(loc_less_3000, by = c("location_id" = "location_id_1")) %>% 
  rename(loc_less_3000m = location_id_2)

# Convert geometry to WKT
places$geometry <- st_as_text(places$geometry)

# save results
write_csv(places, "data/sudan_places.csv")

places <- read_csv("data/places.csv")

places$geometry <- st_as_sfc(places$geometry, wkt = TRUE)

places <- st_sf(places)


# # investigate distances ######################################################
# distance_df <- read_csv("data/distance_between_locations.csv")
# 
# # show histogram of distances
# distance_df %>%
#   ggplot(aes(distance)) +
#   geom_histogram() +
#   labs(title = "Histogram of distances between locations",
#        x = "Distance (meters)",
#        y = "Frequency")
# 
# # show mean, median, and standard deviation of distances
# distance_df %>%
#   summarise(mean_distance = mean(distance),
#             median_distance = median(distance),
#             sd_distance = sd(distance))
# 
# # show number of instances where distance less than 3000, 2000, 1500, 1000 meters
# distance_df %>%
#   summarise(n_6000 = sum(distance < 6000),
#             n_5000 = sum(distance < 5000),
#             n_4000 = sum(distance < 4000),
#             n_3000 = sum(distance < 3000),
#             n_2500 = sum(distance < 2500),
#             n_2000 = sum(distance < 2000),
#             n_1500 = sum(distance < 1500),
#             n_1000 = sum(distance < 1000))

# # try out buffers ###############################################################

## create admin2 bounding box ####
# find admin2 of Kyiv
admin2_val <- "Polohivskyi"

average_location <- df %>%
  # filter(admin2 == admin2_val) %>%
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
admin2_bbox <- st_bbox(sf_df)
# rename bbox columns
names(admin2_bbox) <- c("left", "bottom", "right", "top")

# get basemap
basemap <- get_stadiamap(
  bbox = admin2_bbox,
  zoom = 7,
  map_type = "toner-lite"
)

## create municipality buffer ####
# extract coordinates of all locations in admin2
municip <- df %>%
  filter(geo_precision ==1) %>%
  select(longitude, latitude) %>%
  unique()

# Convert to sf object and transform CRS if necessary
municip_sf <- st_as_sf(municip, coords = c("longitude", "latitude"), crs = 4326)

# I dont think I need to transform it, because 4326 seems to work fine

# # Transform to a projected CRS for buffering
# municip_sf_proj <- st_transform(municip_sf, crs = 4326) # Using Web Mercator for example

municip_buffer <- st_buffer(municip_sf,
                            dist = 3000,
                            joinStyle = "ROUND")

# # Transform back to WGS 84 for plotting
# municip_buffer <- st_transform(municip_buffer, crs = 4326)


## plot the buffers ############################################################
ggmap(basemap) +
  geom_sf(data = municip_buffer,
          fill = "blue",
          color = "blue",
          alpha = 0.2,
          inherit.aes = FALSE) +
  theme_void()
