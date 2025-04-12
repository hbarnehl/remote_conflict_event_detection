import pandas as pd

import getpass

from planet import Session, Auth
from code.acquire_data.planet_helpers import parse_polygon, main_order, repair_aoi, get_offline_order_names, download_order_metadata

import asyncio
from tqdm.asyncio import tqdm  # Use tqdm.asyncio for async support

import logging


async def process_download(DOWNLOAD_DIR, search, cl, pbar, semaphore):
    async with semaphore:
        try:
            await main_order(DOWNLOAD_DIR, search, cl, download=False)
            
        except Exception as e:
            error_message = str(e)
            if "AOI is invalid" in error_message:
                logging.error(f"{search['name']}: {e}\n\t Trying to repair AOI")
                try:
                    search["geom"]["coordinates"][0] = repair_aoi(search["geom"]["coordinates"][0])
                    await main_order(DOWNLOAD_DIR, search, cl, download=False)
                except Exception as e:
                    logging.error(f"{search['name']}: Could not repair AOI.{e}")
        finally:
            pbar.update(1)

def create_order_list(df):
        order_list = []

        # iterate over rows of df
        for index, row in df.iterrows():
            before_item_ids = row["before_image_id"]
            before_item_ids.reverse()
            after_item_ids = row["after_image_id"]
            after_item_ids.reverse()
            coords = parse_polygon(row["geometry"])
            geom = {
                "type": "Polygon",
                "coordinates": coords
            }

            before_order_dict = {
                "name" : row["search_id"] + "_before",
                "item_ids" : before_item_ids,
                "geom": geom
            }

            after_order_dict = {
                "name" : row["search_id"] + "_after",
                "item_ids" : after_item_ids,
                "geom": geom
            }

            order_list.append(before_order_dict)
            order_list.append(after_order_dict)
        return order_list

def prepare_data(sample_df, places_df):
    # merge both dfs on location_id
    df = sample_df.merge(places_df, on="location_id")

    df["after_image_id"] = df["after_image_id"].str.strip("[]").str.replace("'", "").str.split(", ")
    df["before_image_id"] = df["before_image_id"].str.strip("[]").str.replace("'", "").str.split(", ")

    random_seed = 42

    # sample 1000 rows where search_id starts with "events"
    events_sample = df[df["search_id"].str.startswith("events")]

    # sample 1000 rows where search_id starts with "non_events"
    non_events_sample = df[df["search_id"].str.startswith("non_events")]

    # combine both samples in one df
    output_df = pd.concat([events_sample, non_events_sample])

    return output_df

async def main():
    user = input("Username: ")
    pw = getpass.getpass()
    auth = Auth.from_login(user,pw)
    auth.store()

    DATA_DIR = "../data"
    SEARCH_DIR = DATA_DIR + "/searches"
    DOWNLOAD_DIR = DATA_DIR + "/images_sudan"
    ORDER_DIR = DATA_DIR + "/orders_sudan"

    logging.basicConfig(
    filename='../download_images.log',  # Log file location
    level=logging.ERROR,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
    )

    sample_df = pd.read_csv(DATA_DIR + "/sudan_download_sample.csv")

    places_df = pd.read_csv(DATA_DIR + "/sudan_places.csv")[["location_id", "geometry"]]

    sample = prepare_data(sample_df, places_df)
    
    order_list = create_order_list(sample)
    print(f"Number of orders from sample: {len(order_list)}")

    # update list of processed orders
    await download_order_metadata(ORDER_DIR)

    offline_order_names = get_offline_order_names(ORDER_DIR)

    if offline_order_names:
        order_list = [order for order in order_list if order["name"] not in offline_order_names]

    if len(order_list) == 0:

        print("All orders are already processed")
        return
    
    else:
        print(f"{len(order_list)} orders not in offline")
        semaphore = asyncio.Semaphore(200)  
        async with Session() as sess:
            cl = sess.client('orders')

            existing_orders = [o async for o in cl.list_orders()]

            # only keep orders with names not already in existing orders
            order_list = [order for order in order_list if order["name"] not in [o["name"] for o in existing_orders]]
            print(f"{len(order_list)} orders still to download")

            pbar = tqdm(total=len(order_list), smoothing=0.05)
            await asyncio.gather(*(process_download(DOWNLOAD_DIR, search, cl, pbar, semaphore) for search in order_list))
            pbar.close()

if __name__ == "__main__":
    asyncio.run(main())