library(tidyverse)
library(sf)


df <- read_csv("data/ACLED_Ukraine_2013-11-01-2024-12-16.csv")
places <- read_csv("data/places.csv")

# Create events timeline ######################################################

# filter df
df_filtered <- df %>% 
  filter(event_type == "Explosions/Remote violence",
         sub_event_type %in% c("Air/drone strike",
                               "Shelling/artillery/missile attack"),
         geo_precision == 1,
         time_precision == 1)


# are there locations that have more than one pair of coordinates?
# df_filtered %>% 
#   group_by(location, latitude, longitude) %>% 
#    count() %>% 
#   View()
# 
# df_filtered %>% 
#   filter(admin3 == "Alchevska",
#          latitude == 48.3234,
#          longitude == 38.5227) %>% 
#   View("Alchevska")


# seems like the ones that do are actually 200different places
# with the same name


# use lubridate to change date to date format
df_filtered <- df_filtered %>% 
  mutate(event_date = dmy(event_date))

days <- seq.Date(min(df_filtered$event_date), max(df_filtered$event_date), by="day")

# create new data frame with all combinations of places and days
df_all <- expand_grid(places, days) %>% 
  rename(event_date = days) %>% 
  select(location_id, event_date, loc_less_3000m)

# left join the filtered data with the all data
df_all_joined <- df_all %>% 
  left_join(df_filtered, by=c("location_id", "event_date"))

# remove unneeded columns
df_all_joined <- df_all_joined %>% 
  select(-c(geo_precision, time_precision, source_scale, source, iso,
            inter1, inter2, interaction, disorder_type, tags, timestamp,
            country, admin1, admin2, admin3, location, latitude, longitude,
            actor1, actor2, assoc_actor_1, assoc_actor_2, region, notes,
            fatalities, civilian_targeting, event_type, sub_event_type, year))


# create a new column that is 1 if there was an event and 0 if there wasn't
df_all_joined <- df_all_joined %>% 
  mutate(event = ifelse(is.na(event_id_cnty), 0, 1))

# create a new column that is 1 if there was an event in a location less than
# 3000 m away
df_all_joined_1 <- df_all_joined %>%
  mutate(loc_less_3000m = str_split(loc_less_3000m, ",\\s*")) %>%
  unnest(loc_less_3000m) %>%
  mutate(loc_less_3000m = as.integer(loc_less_3000m)) %>% 
  left_join(df_all_joined %>%
              select(location_id, event_date, event) %>% 
              rename(event_2 =event), 
            by = c("loc_less_3000m" = "location_id", "event_date")) %>% 
  group_by(location_id, event_date) %>%
  summarise(overlapping_event = as.integer(any(event_2 == 1, na.rm = TRUE)), .groups = 'drop') %>%
  right_join(df_all_joined, by = c("location_id", "event_date")) %>%
  mutate(overlapping_event = replace_na(overlapping_event, 0))

# create new binary column "any_event" that is 1 if either event or overlapping_event is 1
df_all_joined_1 <- df_all_joined_1 %>% 
  mutate(any_event = ifelse(event == 1 | overlapping_event == 1, 1, 0))

# Extract positives and negatives #############################################


# create binary variable to record if attack is first in a location
df_final <- df_all_joined_1 %>% 
  group_by(location_id) %>%
  arrange(event_date) %>%
  mutate(cum_attack = cumsum(any_event),
         n_attack = ifelse(any_event == 0, NA, cum_attack),
         first_attack = ifelse(n_attack == 1, 1, 0)) %>% 
  ungroup()

# create binary variables to indicate if attack is isolated in 2,4,6,8,10 day window

# Helper function to create event_iso variables
create_attack_iso <- function(df, window_size) {
  df %>%
    mutate(!!paste0("attack_window", window_size) := ifelse(
      event == 1 & 
        rowSums(sapply(1:window_size, function(i) lead(any_event, i) == 0 &
                         lag(any_event, i) == 0)) == window_size, 1, 0
    ))
}

create_non_attack_iso <- function(df, window_size) {
  df %>%
    mutate(!!paste0("non_attack_window", window_size) := ifelse(
      any_event == 0 & 
        rowSums(sapply(1:window_size, function(i) lead(any_event, i) == 0 &
                         lag(any_event, i) == 0)) == window_size, 1, 0
    ))
}

# Apply the helper function for each window size
df_final <- df_final %>%
  group_by(location_id) %>%
  { create_event_iso(., 1) } %>%
  { create_event_iso(., 2) } %>%
  { create_event_iso(., 3) } %>%
  { create_event_iso(., 4) } %>%
  { create_event_iso(., 5) } %>% 
  { create_non_attack_iso(., 1) } %>%
  { create_non_attack_iso(., 2) } %>%
  { create_non_attack_iso(., 3) } %>%
  { create_non_attack_iso(., 4) } %>%
  { create_non_attack_iso(., 5) } %>%
  ungroup()

# remove unnecessary columns
df_final <- df_final %>% 
  select(-c(loc_less_3000m))

# create timeline id
df_final <- df_final %>% mutate(timeline_id = row_number())

# save the data
write_csv(df_final, "data/ACLED_Ukraine_events_timeline.csv")