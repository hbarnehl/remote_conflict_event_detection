library(tidyverse)

df <- read_csv("data/ACLED_Ukraine_2013-11-01-2024-12-16.csv")

# filter df
df_filtered <- df %>% 
  filter(event_type == "Explosions/Remote violence",
         sub_event_type %in% c("Air/drone strike",
                               "Shelling/artillery/missile attack"),
         geo_precision == 1)

# are there locations that have more than one pair of coordinates?
df_filtered %>% 
  group_by(location, latitude, longitude) %>% 
   count() %>% 
  View()

df_filtered %>% 
  filter(admin3 == "Alchevska",
         latitude == 48.3234,
         longitude == 38.5227) %>% 
  View("Alchevska")


# seems like the ones that do are actually different places
# with the same name


# create list of distinct places and of all days from beginning to end of data

places <- df_filtered %>% 
  distinct(location, latitude, longitude) %>% 
  arrange(location) %>% 
  mutate(id = row_number())

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
