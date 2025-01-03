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


def create_request(geom, items, request_name, item_type, asset_type):
   '''
   This function creates an order request for the specified geometry, items, request name, item type, and asset type.
   geom: The geometry of the location of interest. This is used to clip the images to the specified area.
   items: The list of item IDs to be ordered.
   request_name: The name of the order request (for cataloging).
   item_type: The type of item to be ordered, e.g. PSScene
   asset_type: The type of asset to be ordered, e.g. ortho_visual, ortho_analytic_4b 
   '''
   order = order_request.build_request(
       name=request_name,
       products=[
           order_request.product(item_ids=items,
                                        product_bundle=asset_type,
                                        item_type=item_type)
       ],
       tools=[order_request.clip_tool(aoi=geom)])

   return order

async def create_and_download_order(client, order_detail, directory):
   with reporting.StateBar(state='creating') as reporter:
       order = await client.create_order(order_detail)
       reporter.update(state='created', order_id=order['id'])
       await client.wait(order['id'], callback=reporter.update_state)

   await client.download_order(order['id'], directory, progress_bar=True)

async def main_order(geom, items, request_name, item_type, asset_type):
   async with Session() as sess:
       cl = sess.client('orders')

       # Create the order request
       request = create_request(geom, items, request_name, item_type, asset_type)

       # Create and download the order
       order = await create_and_download(cl, request, DOWNLOAD_DIR)


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
        with open(os.path.join(folder_path, file_name), 'r') as file:
            data = json.load(file)
            json_data.append(data)

    if len(json_data) == 1:
        json_data = json_data[0]

    return json_data

# Example usage:
# folder_path = '../data/searches'
# num_files = 10
# start_index = 5
# timeline_ids = ['events_12345', 'non-events_67890']
# data = load_json_files(folder_path, num_files, timeline_ids, start_index)
# print(data)