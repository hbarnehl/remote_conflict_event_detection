'''
This script moves all directories that end with "after" to "images_ukraine_extracted_after" and renames "images_ukraine_extracted" to "images_ukraine_extracted_before".
It then moves all directories that are not in "images_ukraine_extracted_after" to "images_ukraine_unmatched".
'''

import os
import pandas as pd
from tqdm import tqdm

DATA_DIR = '../data'
IMG_DIR = os.path.join(DATA_DIR, 'images_ukraine_extracted')
IMG_DIR_BEFORE = os.path.join(DATA_DIR, 'images_ukraine_extracted_before')
IMG_DIR_AFTER = os.path.join(DATA_DIR, 'images_ukraine_extracted_after')

# collect all event ids from image files
img_dirs = os.listdir(IMG_DIR)

# move all directories that end with "after" to "images_ukraine_extracted_after"
after_dirs = [img_dir for img_dir in img_dirs if img_dir.endswith('after')]

for img_dir in tqdm(after_dirs):
    os.rename(os.path.join(IMG_DIR, img_dir), os.path.join(IMG_DIR + '_after', img_dir))

# rename IMG_DIR to before
os.rename(IMG_DIR, os.path.join(DATA_DIR, 'images_ukraine_extracted_before'))

img_dirs_before = os.listdir(IMG_DIR_BEFORE)
image_event_ids_before = [img_dir.split('_')[-2] for img_dir in img_dirs_before]
img_dirs_after =  os.listdir(IMG_DIR_AFTER)
image_event_ids_after = [img_dir.split('_')[-2] for img_dir in img_dirs_after]

# list all ids in before that are not in after
ids_not_in_after = [event_id for event_id in image_event_ids_before if event_id not in image_event_ids_after]
ids_not_in_before = [event_id for event_id in image_event_ids_after if event_id not in image_event_ids_before]

# move unmatched directories to new directory
for img_id in tqdm(ids_not_in_after):
    # find img dir belonging to event id
    img_dir = [img_dir for img_dir in img_dirs_before if img_dir.split("_")[-2]==img_id][0]
    os.rename(os.path.join(IMG_DIR_BEFORE, img_dir),
              os.path.join(DATA_DIR, 'images_ukraine_unmatched', img_dir))
    
# now the same for after
for img_id in tqdm(ids_not_in_before):
    # find img dir belonging to event id
    img_dir = [img_dir for img_dir in img_dirs_after if img_dir.split("_")[-2]==img_id][0]
    os.rename(os.path.join(IMG_DIR_AFTER, img_dir),
              os.path.join(DATA_DIR, 'images_ukraine_unmatched', img_dir))
    
############ Now strip dir names of everything but event id ############

img_dirs_before = os.listdir(IMG_DIR_BEFORE)
image_event_ids_before = [int(img_dir.split('_')[-2]) for img_dir in img_dirs_before]
img_dirs_after =  os.listdir(IMG_DIR_AFTER)
image_event_ids_after = [int(img_dir.split('_')[-2]) for img_dir in img_dirs_after]

# strip img dirs of stuff that is not the event id
for img_dir, event_id in tqdm(zip(img_dirs_before, image_event_ids_before)):
    os.rename(os.path.join(IMG_DIR_BEFORE, img_dir),
              os.path.join(IMG_DIR_BEFORE, f'{event_id}'))
# now same for after
for img_dir, event_id in tqdm(zip(img_dirs_after, image_event_ids_after)):
    os.rename(os.path.join(IMG_DIR_AFTER, img_dir),
              os.path.join(IMG_DIR_AFTER, f'{event_id}'))
    
############ Now create annotations file ############

img_dirs_before = [int(id) for id in os.listdir(IMG_DIR_BEFORE)]

event_df = pd.read_csv(os.path.join(DATA_DIR, 'events_data/ACLED_Ukraine_events_timeline.csv'))

# keep only rows where timeline_id is in image_event_ids_before
event_df_filtered = event_df[event_df['timeline_id'].isin(img_dirs_before)]

# left join with places_ukraine.csv
places_df = pd.read_csv(os.path.join(DATA_DIR, 'events_data/ukraine_places.csv'))[['location_id', 'location', "admin1"]]
event_df_filtered = event_df_filtered.merge(places_df, how='left', left_on='location_id', right_on='location_id')

# keep only columns that are relevant
event_df_filtered = event_df_filtered[['timeline_id', 'location_id', "location", "admin1", 'event_date', 'overlapping_event',
                                       'event', 'any_event', 'cum_attack']]
# save to csv
event_df_filtered.to_csv(os.path.join(DATA_DIR, 'annotations_ukraine.csv'), index=False)

import json

img_dirs_before = os.listdir(IMG_DIR_BEFORE)
img_dirs_after =  os.listdir(IMG_DIR_AFTER)

df_columns = ['clear_percent', 'cloud_percent', 'shadow_percent', 'snow_ice_percent','visible_percent']

# empty df
df = pd.DataFrame()
counter=0
for id in tqdm(img_dirs_before):
    counter += 1
    with open(os.path.join(IMG_DIR_BEFORE, id, "files", "composite_metadata.json"), 'r') as file:
        before_json = json.load(file)["properties"]
    with open(os.path.join(IMG_DIR_AFTER, id, "files", "composite_metadata.json"), 'r') as file:
        after_json = json.load(file)["properties"]
    # create new row with the values from the json files, prefix with "before_" and "after_"
    new_row_before = {f'before_{key}': before_json[key] for key in df_columns}
    new_row_after = {f'after_{key}': after_json[key] for key in df_columns}
    # create new row with the id
    new_row = {'id': id}
    # concatenate the new rows
    new_row.update(new_row_before)
    new_row.update(new_row_after)
    # append the new row to the df
    df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

# save to csv
df.to_csv(os.path.join(DATA_DIR, 'metadata_ukraine.csv'), index=False)