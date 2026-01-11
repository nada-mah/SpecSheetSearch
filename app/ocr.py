from paddleocr import PaddleOCR

import numpy as np

def get_ocr_object_per_page(images,ocr):
  ocr_results =[]
  for image in images:
    np_img = np.asarray(image)
    res = ocr.predict(np_img)
    if res:
      ocr_results.append(res)
  return ocr_results

import logging
import json
import re

def build_full_ocr_text(ocr_results):
    big_text = ''
    for i in range(len(ocr_results)):
        rec_texts = ocr_results[i][0]['rec_texts']
        big_text += " ".join(rec_texts).lower()
    logging.debug(f"Built full OCR text (length: {len(big_text)} characters)")
    return big_text

def filter_ocr_key_hit_by_value_matched(ocr_key_hit, value_matched):
    """
    Return ALL OCR hits whose keys appear in value_matched.
    """
    # Build a set of keys we care about (from value_matched's original_key)
    keys_of_interest = {
        vm_data["original_key"] for vm_data in value_matched.values()
    }

    # Collect all OCR hits that match any of these keys
    filtered = [
        hit for hit in ocr_key_hit
        if hit.get("key") in keys_of_interest
    ]

    return filtered

def filter_ocr_keys_by_regions(ocr_key_hits, regions_by_page):
    """
    Filter OCR key hits by a list of region dicts grouped by page index.
    Adds 'stop_y' to each hit, which is the bottom Y coordinate of the overlapping region.

    Args:
        ocr_key_hits: list of dicts, each with 'ocr_result_index' and 'bbox'
        regions_by_page: dict mapping page_index -> list of regions
            Each region is {'x1', 'y1', 'x2', 'y2'}

    Returns:
        list of OCR key hits that overlap at least one region on their page,
        with an extra field 'stop_y' = region['y2']
    """

    def poly_to_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def boxes_intersect(a, b):
        return not (
            a[2] < b[0] or
            a[0] > b[2] or
            a[3] < b[1] or
            a[1] > b[3]
        )

    filtered = []

    for hit in ocr_key_hits:
        page_idx = hit.get("ocr_result_index")
        poly = hit.get("bbox")
        if poly is None or page_idx not in regions_by_page:
            continue

        key_box = poly_to_bbox(poly)

        # check if overlaps any region on this page
        for region in regions_by_page[page_idx]:
            region_box = (region['x1'], region['y1'], region['x2'], region['y2'])
            if boxes_intersect(key_box, region_box):
                # add stop_y to the hit
                hit_copy = hit.copy()
                hit_copy['stop_y'] = region['y2']
                filtered.append(hit_copy)
                break  # one match is enough

    return filtered
  
def match_values_for_keys(
    row_for_key_data,
    value_matched,
    value_not_matched
):
    """
    Match OCR-extracted text under each key against allowed values
    and split into matched / not matched dictionaries.
    """

    if value_not_matched is None:
        value_not_matched = {}

    # Build lookup: key -> combined OCR text
    ocr_text_by_key = {}
    for row in row_for_key_data:
        key = row["key"]
        combined_text = " ".join(
            t["text"] for t in row.get("text", [])
        ).lower()
        ocr_text_by_key[key] = combined_text

    final_value_matched = {}
    final_value_not_matched = value_not_matched.copy()

    for attr_name, attr_obj in value_matched.items():
        key = attr_obj.get("original_key") or attr_name
        values = attr_obj.get("values", {})

        combined_text = ocr_text_by_key.get(key, "")
        new_values = {}
        any_value_hit = False

        for value in values.keys():
            value_norm = value.strip().lower()

            if value_norm and value_norm in combined_text:
                new_values[value] = True
                any_value_hit = True
            else:
                new_values[value] = False

        new_attr = attr_obj.copy()
        new_attr["values"] = new_values

        if any_value_hit:
            final_value_matched[attr_name] = new_attr
        else:
            final_value_not_matched[attr_name] = new_attr

    return final_value_matched, final_value_not_matched
