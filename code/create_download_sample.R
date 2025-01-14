library(tidyverse)



timeline_df <- read_csv("data/ACLED_Ukraine_events_timeline.csv")

search_df <- read_csv("data/best_search_results.csv")


# create column timeline_id, which is the numbers of the column search_id
search_df <- search_df %>% 
  mutate(timeline_id = as.numeric(str_extract(search_id, "\\d+"))) %>% 
  filter(!is.na(after_image_id) & !is.na(before_image_id))


# merge the two dataframes
merged_df <- search_df %>% 
  left_join(timeline_df, by = c("timeline_id"))

merged_df <- merged_df %>% 
  rowwise() %>%
  mutate(lowest_clear=min(before_agg_clear,after_agg_clear)) %>% 
  # keep only where before_overlap and after_overlap are 1
  filter(before_overlap == 1 & after_overlap == 1,
         lowest_clear >= 90)

# create column attack window showing max window available for row
merged_df <- merged_df %>% 
  mutate(attack_window = attack_window1 + attack_window2 + attack_window3 + attack_window4 + attack_window5,
         non_attack_window = non_attack_window1 + non_attack_window2 +non_attack_window3+non_attack_window4+non_attack_window5) %>% 
  rowwise() %>% 
  mutate(# create column recording the max distance in days between the before/after images and the event date
         max_distance = max(abs(difftime(before_date, event_date, units = "days")), abs(difftime(after_date, event_date, units = "days"))),
         # create column recording whether before and after images are taken within isolated window
         within_attack_window = if_else(max_distance <= attack_window, 1, 0),
         within_non_attack_window = if_else(max_distance <= non_attack_window, 1,0)) %>% 
  # keep only rows where within window
  filter(within_attack_window == 1 | within_non_attack_window == 1)

# keep columns search_id, before_image_id, after_image_id, geometry
merged_df <- merged_df %>% 
  select(search_id, before_image_id, after_image_id)


# save the dataframe
write_csv(merged_df, "data/download_sample.csv")
```