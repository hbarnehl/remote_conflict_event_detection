library(tidyverse)

df <- read_csv("data/ACLED_Ukraine_2013-11-01-2024-12-16.csv")

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


# create list of distinct places and of all days from beginning to end of data

places <- df_filtered %>% 
  distinct(location, latitude, longitude) %>% 
  arrange(location) %>% 
  mutate(location_id = row_number())

# use lubridate to change date to date format
df_filtered <- df_filtered %>% 
  mutate(event_date = dmy(event_date))

days <- seq.Date(min(df_filtered$event_date), max(df_filtered$event_date), by="day")

# create new data frame with all combinations of places and days
df_all <- expand_grid(places, days) %>% 
  rename(event_date = days)

# left join the filtered data with the all data
df_all_joined <- df_all %>% 
  left_join(df_filtered, by=c("location", "latitude", "longitude", "event_date"))


# create a new column that is 1 if there was an event and 0 if there wasn't
df_all_joined <- df_all_joined %>% 
  mutate(event = ifelse(is.na(event_id_cnty), 0, 1))

# remove unneeded columns
df_all_joined <- df_all_joined %>% 
  select(-c(geo_precision, time_precision, source_scale, source, iso,
            inter1, inter2, interaction, disorder_type, tags, timestamp,
            country))


# Extract positives and negatives #############################################


# create binary variable to record if attack is first in a location
df_final <- df_all_joined %>% 
  group_by(location_id) %>%
  arrange(event_date) %>%
  mutate(cum_attack = cumsum(event),
         n_attack = ifelse(event == 0, NA, cum_attack),
         first_attack = ifelse(n_attack == 1, 1, 0)) %>% 
  ungroup()

# create binary variables to indicate if attack is isolated in 2,4,6,8,10 day window

# Helper function to create event_iso variables
create_event_iso <- function(df, window_size, attack) {
  name <- ifelse(attack == 1, "attack_window", "non_attack_window")
  df %>%
    mutate(!!paste0(name, window_size) := ifelse(
      event == attack & 
        rowSums(sapply(1:window_size, function(i) lead(event, i) == 0 & lag(event, i) == 0)) == window_size,
      1, 0
    ))
}

# Apply the helper function for each window size
df_final <- df_final %>%
  group_by(location_id) %>%
  { create_event_iso(., 1, attack=1) } %>%
  { create_event_iso(., 2, attack=1) } %>%
  { create_event_iso(., 3, attack=1) } %>%
  { create_event_iso(., 4, attack=1) } %>%
  { create_event_iso(., 5, attack=1) } %>% 
  { create_event_iso(., 1, attack=0) } %>%
  { create_event_iso(., 2, attack=0) } %>%
  { create_event_iso(., 3, attack=0) } %>%
  { create_event_iso(., 4, attack=0) } %>%
  { create_event_iso(., 5, attack=0) } %>%
  ungroup()

# save the data
write_csv(df_final, "data/ACLED_Ukraine_events_timeline.csv")