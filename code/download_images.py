import pandas as pd
import os
import json

import getpass

from planet import Session, Auth

import asyncio
from tqdm.asyncio import tqdm  # Use tqdm.asyncio for async support

import logging


async def process_download(order_file, download_dir, cl, pbar, semaphore):
    async with semaphore:
        try:
            # read order json and extract order id
            with open(order_file, 'r') as f:
                order = json.load(f)

            await cl.download_order(order['id'], download_dir, progress_bar=False, overwrite=False)
            
        except Exception as e:
            logging.error(f"could not download order {order['id']}: {e}")

        finally:
            pbar.update(1)

async def main():
    # user = input("Username: ")
    # pw = getpass.getpass()
    # auth = Auth.from_login(user,pw)
    # auth.store()

    DATA_DIR = "../data"
    DOWNLOAD_DIR = DATA_DIR + "/images_ukraine"
    ORDER_DIR = DATA_DIR + "/orders_ukraine"

    logging.basicConfig(
    filename='../download_images.log',  # Log file location
    level=logging.ERROR,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
    )

    # update list of processed orders
    # await download_order_metadata(ORDER_DIR)

    # create order list: all files in ORDER_DIR
    order_list = os.listdir(ORDER_DIR)[59:]

    if not order_list:
        print("No offline orders found")
        return
    
    print(f"{len(order_list)} orders found")
    semaphore = asyncio.Semaphore(1)  
    async with Session() as sess:
        cl = sess.client('orders')

        pbar = tqdm(total=len(order_list), smoothing=0.05)
        await asyncio.gather(*(process_download(ORDER_DIR+ "/" + order, DOWNLOAD_DIR, cl, pbar, semaphore) for order in order_list))
        pbar.close()

if __name__ == "__main__":
    asyncio.run(main())