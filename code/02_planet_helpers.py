from planet import order_request, Session, reporting, data_filter
from datetime import datetime
import os
import json


def parse_polygon(polygon_str):
    # Remove the 'POLYGON ((' prefix and '))' suffix
    polygon_str = polygon_str.replace('POLYGON ((', '').replace('))', '')
    # Split the string into coordinate pairs
    coord_pairs = polygon_str.split(', ')
    # Convert each coordinate pair into a list of floats
    coordinates = [[float(coord) for coord in pair.split()] for pair in coord_pairs]
    # Wrap the list of coordinates in an additional list to match the desired structure
    return [coordinates]

def repair_aoi(geom):
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    polygon = Polygon(geom)
    if not polygon.is_valid:
        repaired_geom = make_valid(polygon)
        repaired_geom = repaired_geom.geoms[0].exterior.coords[:]
        repaired_geom = [list(pair) for pair in repaired_geom]
        return repaired_geom
    else:
        return geom

def create_request(search_dict):
   '''
   This function creates an order request for the specified geometry, items, request name, item type, and asset type.
   geom: The geometry of the location of interest. This is used to clip the images to the specified area.
   items: The list of item IDs to be ordered.
   request_name: The name of the order request (for cataloging).
   item_type: The type of item to be ordered, e.g. PSScene
   asset_type: The type of asset to be ordered, e.g. ortho_visual, ortho_analytic_4b 
   '''
   order = order_request.build_request(
       name=search_dict["name"],
       products=[
           order_request.product(item_ids=search_dict["item_ids"],
                                        product_bundle="analytic_udm2",
                                        item_type="PSScene")
       ],
       tools=[order_request.clip_tool(aoi=search_dict["geom"]),
              order_request.composite_tool()
              ],
        delivery = order_request.delivery(
            archive_type='zip',
            single_archive=True,
            archive_filename='{{name}}.zip'
            )
        )

   return order

async def create_and_download_order(client, order_detail, directory, download=True):
   with reporting.StateBar(state='creating') as reporter:
       order = await client.create_order(order_detail)
       reporter.update(state='created', order_id=order['id'])
       await client.wait(order['id'], callback=reporter.update_state, max_attempts=600)
   if download:
       await client.download_order(order['id'], directory, progress_bar=True)

async def main_order(DOWNLOAD_DIR, search, cl, download=True):
       # Create the order request
       request = create_request(search)

       # Create and download the order
       order = await create_and_download_order(cl, request, DOWNLOAD_DIR, download)


def set_filters(from_date, to_date, geom):
    '''
    This function sets the filters for the search of images.

    from_date: 'YYYY-MM-DD'; The start date of the search .
    to_date: 'YYYY-MM-DD'; The end date of the search.
    geom: output of parse_polygon; The geometry of the location of interest.
    '''

    # Set the filters for the search
    # geom filter: search for items whose footprint geometry inter-
    # sects with specified geometry
    geom_filter = data_filter.geometry_filter(geom)

    # # only include images with clear_percent greater than 90 
    # clear_percent_filter = data_filter.range_filter('clear_percent', gt = min_clear)

    # only look for images between following dates
    date_range_filter = data_filter.date_range_filter("acquired", gt = from_date, lt=to_date)

    # # cloudcover less than 0.1
    # cloud_cover_filter = data_filter.range_filter('cloud_cover', lt = 0.1)

    # combine individual filters
    combined_filter = data_filter.and_filter([geom_filter, date_range_filter])

    return combined_filter

async def create_and_download_search(name,filter, item_types="PSScene"):
    async with Session() as sess:
        cl = sess.client('data')
        request = await cl.create_search(name=name,
                                        search_filter=filter,
                                        item_types=item_types)
        
    # The limit paramter allows us to limit the number of results from our search that are returned.
    # The default limit is 100. Here, we're setting our result limit to 50.
    async with Session() as sess:
        cl = sess.client('data')
        items = cl.run_search(search_id=request['id'], limit=100)
        item_list = [i async for i in items]
    
    return item_list

def load_search_files(folder_path, num_files, timeline_ids=None, start_index=0):
    files = os.listdir(folder_path)
    json_data = []

    if timeline_ids:
        # Load files by timeline_ids
        files_to_load = [f for f in files if any(timeline_id in f for timeline_id in timeline_ids)]
    else:
        # Load files by alphabetic order with starting index
        files_to_load = sorted(files)[start_index:start_index + num_files]

    for file_name in files_to_load:
        try:
            with open(os.path.join(folder_path, file_name), 'r') as file:
                data = json.load(file)
                json_data.append(data)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            continue
    if len(json_data) == 1:
        json_data = json_data[0]

    return json_data


def get_offline_order_names(directory, files=None):
    import os
    import json
    from tqdm import tqdm
    
    if not files:
        files = os.listdir(directory)
        if len(files) == 0:
            return
    
    order_names = []

    print("parsing files")
    for file in tqdm(files):
        with open(directory+"/"+file, 'r') as f:
            data = json.load(f)
            order_name = data['name']
            order_names.append(order_name)
    
    return order_names

def get_latest_file_creation_date(directory, files=None):
    import os
    import json
    from tqdm import tqdm

    if not files:
        files = os.listdir(directory)
        if len(files) == 0:
            return None
    # Get the file with the latest creation time
    created_on = []

    print("parsing files")
    for file in tqdm(files):
        with open(directory+"/"+file, 'r') as f:
            data = json.load(f)
            creation_time = data['created_on']
            created_on.append(creation_time)
    # get the latest creation time
    creation_time = max(created_on)

    return creation_time

# write function to download the metadata of all orders whose metadata has not been downloaded yet
async def download_order_metadata(ORDER_DIR, creation_time=None):
    import os
    import json
    from tqdm import tqdm
    
    files = os.listdir(ORDER_DIR)
    if not creation_time:
        creation_time = get_latest_file_creation_date(ORDER_DIR, files)
        if not creation_time:
            return None
    
    from_date = creation_time+"/.."

    print(f"downloading orders from {from_date.split('.')[0]}")

    async with Session() as sess:
        try:
            cl = sess.client('orders')
            orders_online = [o async for o in cl.list_orders(limit=0, created_on=from_date)]
        except Exception as e:
            logging.error(f"could not retrieve orders: {e}")
            return
        
        print(f"{len(orders_online)} orders online")
        # filter online orders
        orders_online = [o for o in orders_online if o['id']+".json" not in files]
        print(f"{len(orders_online)} not yet downloaded")

        # download metadata of online orders
        if len(orders_online) > 0:
            print("writing orders to disk")
            for order in orders_online:
                order_id = order['id']
                try:
                    with open(f"{ORDER_DIR}/{order_id}.json", "w") as f:
                        json.dump(order, f, indent=4)
                except Exception as e:
                    logging.error(f"{order_id}: {e}")

