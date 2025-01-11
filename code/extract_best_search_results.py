from planet_helpers import load_search_files
from datetime import datetime
from datetime import timedelta
import pandas as pd

import os 
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from itertools import combinations

import csv
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed



def new_round(number, decimals=0):
    if number is None:
        return None
    else:
        return round(number, decimals)

def append_to_csv(file_path, data, header=None):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header and not file_exists:
            writer.writerow(header)
        writer.writerows(data)

def get_combinations(elements):
    """Generate all non-empty combinations of the given elements sorted by their clear percent value."""
    combins = []
    for r in range(1, min(len(elements) + 1, 4)):
        for combo in combinations(elements, r):
            combo = sorted(combo, key=lambda x: x["properties"].get("clear_percent", 0), reverse=True)
            combins.append(list(combo))
    return combins

def calculate_overlap(geometry, aoi):
    """Calculate the overlap area between a geometry and the area of interest."""
    return geometry.intersection(aoi).area / aoi.area

def calculate_weighted_clear_percentage(combo, aoi):
    """Calculate the weighted average of clear percentage for a combination."""
    visible_area = Polygon()
    total_visible_area = 0.0
    combo_clear_percentage = 0
    weighted_sum = 0
    for result in combo:
        geom = shape(result["geometry"])
        clear_percentage = result["properties"].get("clear_percent", 0)

        # Calculate the intersection of the current polygon with the AOI
        intersection = geom.intersection(aoi)
        
        # Calculate the new visible area by subtracting the already visible area
        new_visible_area = intersection.difference(visible_area)
        
        # Update the visible area
        visible_area = unary_union([visible_area, new_visible_area])
        
        # Calculate the area of the new visible part
        new_visible_area_size = new_visible_area.area
        
        # Update the weighted sum and total visible area
        weighted_sum += new_visible_area_size * clear_percentage
        total_visible_area += new_visible_area_size

    combo_clear_percentage = weighted_sum / total_visible_area

    return combo_clear_percentage

def find_best_combination(day_results, aoi):
    """Find the best combination of results for a given day."""
    best_combination = None
    best_weighted_average = -1
    smallest_size = float('inf')
    best_overlap = None

    for combo in get_combinations(day_results):
        combo_ids = [result["id"] for result in combo]
        combo_geoms = [shape(result["geometry"]) for result in combo]
        combo_union = unary_union(combo_geoms)
        combo_overlap = calculate_overlap(combo_union, aoi)

        if combo_overlap > 0.95:
            combo_clear_percentage = calculate_weighted_clear_percentage(combo, aoi)

            # Check if this combination is better or equally good but smaller
            if (combo_clear_percentage > best_weighted_average) or \
               (combo_clear_percentage == best_weighted_average and len(combo) < smallest_size):
                best_weighted_average = combo_clear_percentage
                best_combination = combo_ids
                smallest_size = len(combo)
                best_overlap = combo_overlap

    return best_combination, best_weighted_average,best_overlap

def find_best(results, event_date, aoi):
    """Find the best image combinations before and after a given event date."""
    aoi_polygon = Polygon(aoi)
    days = set(result["properties"]["acquired"].split("T")[0] for result in results)
    best_day_results = {}

    for day in days:
        day_results = [result for result in results if result["properties"]["acquired"].split("T")[0] == day]
        day_result_geoms = [shape(result["geometry"]) for result in day_results]
        union_geometry = unary_union(day_result_geoms)
        overlap = calculate_overlap(union_geometry, aoi_polygon)

        if overlap > 0.95:
            best_combination, best_weighted_average, calculated_overlap = find_best_combination(day_results, aoi_polygon)
            
            if best_combination:
                best_day_results[day] = (best_combination, best_weighted_average, calculated_overlap)

    best_before = None
    best_after = None
    
    for day in best_day_results:
        if day < event_date:
            if not best_before or best_day_results[day][1] > best_day_results[best_before][1]:
                best_before = day
        elif day > event_date:
            if not best_after or best_day_results[day][1] > best_day_results[best_after][1]:
                best_after = day

    best_before_result = best_day_results.get(best_before, (None, None, None))
    best_after_result = best_day_results.get(best_after, (None, None, None))

    best_results_dict = {
        "best_before": {
            "date": best_before,
            "ids": best_before_result[0],
            "clear_percent": new_round(best_before_result[1], 0),
            "overlap": new_round(best_before_result[2], 0)
        },
        "best_after": {
            "date": best_after,
            "ids": best_after_result[0],
            "clear_percent": new_round(best_after_result[1], 0),
            "overlap": new_round(best_after_result[2], 0)
        }
    }

    return best_results_dict

def process_data(search):
    processed_data = []
    try:
        search_name = search["name"]
        logging.info(f"Processing search {search_name}")
        search_id = search["id"]
        aoi = search["filter"]["config"][0]["config"]["coordinates"][0]
        day_minus_5 = datetime.strptime(search["filter"]["config"][1]["config"]["gt"], "%Y-%m-%dT%H:%M:%SZ")
        event_date = (day_minus_5 + timedelta(days=5)).strftime("%Y-%m-%d")
        quality_results = [result for result in search["results"] if result["properties"]["quality_category"] == "standard"]

        best_results_dict = find_best(quality_results, event_date, aoi)
        processed_data.append([search_name,
                               best_results_dict["best_before"]["date"], best_results_dict["best_before"]["ids"], best_results_dict["best_before"]["clear_percent"], best_results_dict["best_before"]["overlap"],
                               best_results_dict["best_after"]["date"], best_results_dict["best_after"]["ids"], best_results_dict["best_after"]["clear_percent"], best_results_dict["best_after"]["overlap"]])
    except Exception as e:
        logging.error(f"Error processing search {search['id']}: {e}")

    return processed_data

def main():
    logging.basicConfig(
        filename='../search_results_process.log',  # Log file location
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
    )


    DATA_DIR = "../data"
    SEARCH_DIR = DATA_DIR + "/searches"
    OUTPUT_CSV = DATA_DIR + "/best_search_results.csv"

    # check if output_csv exists
    if os.path.exists(OUTPUT_CSV):
        # load existing CSV file
        best_results_df = pd.read_csv(OUTPUT_CSV)
        header = None
    else:
        header = ["search_id", "before_date", "before_image_id", "before_agg_clear", "before_overlap",
                "after_date", "after_image_id", "after_agg_clear", "after_overlap"]

    # create best matches for each search with columns search_id, before_image_id, before_cloud_cover/clarity etc., after_image_id, after_cloud_cover
    start_index = 0
    chunk_size = 10000

    while True:
        json_data = load_search_files(SEARCH_DIR, chunk_size, start_index=start_index)
        if not json_data:
            print("No more data to process")
            break

        if header is None:
            json_data = [search for search in json_data if search["name"] not in best_results_df["search_id"].values]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_data, search): search for search in json_data}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Processing Searches", unit="search"):
                processed_data = future.result()
                append_to_csv(OUTPUT_CSV, processed_data, header=header)

        start_index += chunk_size
        

if __name__ == "__main__":
    main()