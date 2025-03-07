import pandas as pd
import asyncio

import getpass
from planet import Auth, Session
import json
from planet_helpers import set_filters, parse_polygon, load_search_files
from datetime import datetime
import os
import logging
from aiofiles import open as aio_open
from tqdm.asyncio import tqdm  # Use tqdm.asyncio for async support

import logging
logging.basicConfig(
    filename='../search_download.log',  # Log file location
    level=logging.ERROR,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

async def process_file(dir, cl, pbar, semaphore):
    async with semaphore:
        try:
            file_path = os.path.join(SEARCH_DIR, dir)
            
            async with aio_open(file_path, "r+") as f:
                search = json.loads(await f.read())
                if "results" in search.keys():
                    return

                # Run the search and collect results
                items = cl.run_search(search_id=search['id'], limit=100)
                item_list = [i async for i in items]

                # Update the search dictionary with results
                search["results"] = item_list

                # Move the file pointer to the beginning
                await f.seek(0)

                # Write the updated JSON data
                await f.write(json.dumps(search, indent=4))

                # Truncate the file to remove any leftover data
                await f.truncate()
        except Exception as e:
            logging.error(f"processing file {dir}: {e}")
        finally:
            pbar.update(1)

async def process_files(files, cl, pbar, semaphore):
    # Process files concurrently with semaphore
    await asyncio.gather(*(process_file(dir, cl, pbar, semaphore) for dir in files))

def chunk_files(file_list, chunk_size):
    for i in range(0, len(file_list), chunk_size):
        yield file_list[i:i + chunk_size]

async def main():
    all_files = os.listdir(SEARCH_DIR)
    semaphore = asyncio.Semaphore(50)  # Limit concurrency to 10
    async with Session() as sess:
        cl = sess.client('data')

        pbar = tqdm(total=len(all_files), smoothing=0.05)
        await process_files(all_files, cl, pbar, semaphore)
        pbar.close()

        # for file_chunk in chunk_files(all_files, 10000):
        #     pbar = tqdm(total=len(file_chunk), smoothing=0.1)
        #     await process_files(file_chunk, cl, pbar, semaphore)
        #     pbar.close()

if __name__ == "__main__":
    DOWNLOAD_DIR = '../data/planet_data'
    FILTER_DIR = '../data/filters_sudan'  
    SEARCH_DIR = "../data/searches_sudan"

    user = input("Username: ")
    pw = getpass.getpass()
    auth = Auth.from_login(user,pw)
    auth.store()

    asyncio.run(main())