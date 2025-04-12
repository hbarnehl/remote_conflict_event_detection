library(tidyverse)
source("code/helpers.R")

df <- read_csv("data/ACLED_Ukraine_events_timeline.csv")

# create sample of non-events
# to make things easy, take only non-events with where non-attack-window5 is 1
df_non_events_before_attack <- df %>%
  filter(non_attack_window5 == 1,
         cum_attack == 0) %>% 
  slice_sample(n=25000)

df_non_events_after_attack <- df %>%
  filter(non_attack_window5 == 1,
         cum_attack > 0) %>% 
  slice_sample(n=25000)

df_non_events <- bind_rows(df_non_events_before_attack, df_non_events_after_attack)

df_events <- df %>%
  filter(any_event == 1)
  
# save results
write_csv(df_non_events, "data/ACLED_Ukraine_non_events_sample.csv")
write_csv(df_events, "data/ACLED_Ukraine_events_sample.csv")
