import logging
import re
from app.helper import generate_ocr_variants
from app.input_handler import get_attribute_info_by_key

def matches_key_value_pair(big_text: str, key: str, value) -> bool:
    """
    Check if the big OCR text contains a key-value pair matching the given key and value(s),
    including OCR-tolerant variants.

    Args:
        big_text (str): The concatenated OCR text to search in.
        key (str): The expected key (e.g., "CCT").
        value (str or List[str]): A single value or list of possible values (e.g., ["2700K", "3000K"]).

    Returns:
        bool: True if any value or its OCR variants matches the key in the expected pattern.
    """
    logging.debug(f"Checking for key-value pair: key='{key}', value(s)={value}")

    if isinstance(value, str):
        value = [value]

    # Normalize big_text once
    big_text_lower = big_text.lower()

    # Escape and clean the key once
    key_clean = key.strip().lower()
    if not key_clean:
        logging.debug("Key is empty after cleaning – skipping match check.")
        return False
    key_esc = re.escape(key_clean)

    # Common delimiters between key and value
    delimiters = r"[:\-\–\—=]?\s*"

    for val in value:
        val_clean = str(val).strip()
        if not val_clean:
            continue

        # Generate OCR-friendly variants for the value
        ocr_variants = generate_ocr_variants(val_clean)
        logging.debug(f"Generated {len(ocr_variants)} OCR variants for value '{val_clean}': {ocr_variants}")

        for variant in ocr_variants:
            val_esc = re.escape(variant.lower())
            pattern = rf"{key_esc}{delimiters}{val_esc}"
            
            if re.search(pattern, big_text_lower, re.IGNORECASE):
                logging.debug(f"✅ Match found! Pattern '{pattern}' detected in text.")
                return True

    logging.debug(f"❌ No match found for key '{key}' with any of the provided values or their OCR variants.")
    return False

def find_hits(big_text: str, search_terms):
    """
    Searches for occurrences of items in `search_terms` within `big_text`,
    including OCR-tolerant variants.
    Returns a list of original terms that had at least one matching variant in the text.
    """
    big_text_lower = big_text.lower()
    hits = []
    total_terms = len(search_terms)

    logging.debug(f"Searching for {total_terms} term(s) in OCR text (length: {len(big_text)})")

    for term in search_terms:
        term_hits = False
        for variant in generate_ocr_variants(term):
            if variant in big_text_lower:
                hits.append(term)
                term_hits = True
                logging.debug(f"✅ Match found for term '{term}' via variant '{variant}'")
                break  # stop after first successful variant

        if not term_hits:
            logging.debug(f"❌ No match found for term: '{term}'")

    logging.debug(f"Total hits found: {len(hits)} / {total_terms}")
    return hits


def find_key_hits_from_ocr(keys, ocr_results):
    """
    keys: iterable of attribute keys (strings)
    ocr_results: OCR output list

    returns: set of original keys that had an exact word hit
    """
    logging.debug(f"Searching for {len(keys)} key(s) in OCR results...")

    # Normalize keys: tuple of words -> original key
    normalized_key_map = {
        tuple(re.findall(r"\b\w+\b", key.lower())): key
        for key in keys
    }
    matched_keys = set()

    for result_group in ocr_results:
        if not result_group:
            continue
        ocr_result = result_group[0]
        rec_texts = ocr_result.get("rec_texts", [])
        for txt in rec_texts:
            words = re.findall(r"\b\w+\b", txt.lower())
            for key_words, orig_key in normalized_key_map.items():
                key_len = len(key_words)
                # sliding window exact match
                for i in range(len(words) - key_len + 1):
                    if tuple(words[i:i + key_len]) == key_words:
                        if orig_key not in matched_keys:
                            logging.debug(f"✅ Matched key: '{orig_key}' in OCR text: '{txt}'")
                        matched_keys.add(orig_key)
                        break

    logging.debug(f"Key search complete. Found {len(matched_keys)} matching key(s): {sorted(matched_keys)}")
    return matched_keys

def match_product_types_via_lookup(big_text, lookup):
    logging.debug(f"Matching product types against {len(lookup)} lookup entries...")
    matched_product_types = {
        ptype
        for ptype, val in lookup.items()
        if find_hits(big_text, val)
    }
    logging.debug(f"Matched {len(matched_product_types)} product type(s): {sorted(matched_product_types)}")
    return matched_product_types


def split_schema_by_product_type_match(schema, matched_product_types):
    logging.info(f"Filtering schema by {len(matched_product_types)} matched product type(s)...")
    matched = {}
    not_matched = {}

    for attr_name, attr_obj in schema.items():
        original_product_types = attr_obj.get("product_types", [])
        product_type_status = {
            ptype: ptype in matched_product_types
            for ptype in original_product_types
        }

        new_obj = attr_obj.copy()
        new_obj["product_types"] = product_type_status

        has_match = any(product_type_status.values())

        if has_match:
            matched[attr_name] = new_obj
        else:
            new_obj["values"] = {v: False for v in attr_obj.get("values", [])}
            not_matched[attr_name] = new_obj

    logging.debug(f"Schema split: {len(matched)} matched attributes, {len(not_matched)} unmatched.")
    return matched, not_matched


def find_key_hits_from_ocr(keys, ocr_results):
    logging.debug(f"Searching for {len(keys)} keys in OCR results using exact word matching...")

    normalized_key_map = {
        tuple(re.findall(r"\b\w+\b", key.lower())): key
        for key in keys
    }

    matched_keys = set()
    ocr_key_hit = []

    for ocr_idx, result_group in enumerate(ocr_results):
        if not result_group:
            continue

        ocr_result = result_group[0]
        rec_texts = ocr_result.get("rec_texts", [])
        rec_polys = ocr_result.get("rec_polys", [])

        for txt_idx, txt in enumerate(rec_texts):
            words = re.findall(r"\b\w+\b", txt.lower())

            for key_words, orig_key in normalized_key_map.items():
                key_len = len(key_words)

                for i in range(len(words) - key_len + 1):
                    if tuple(words[i:i + key_len]) == key_words:
                        if orig_key not in matched_keys:
                            logging.debug(f"✅ Key matched: '{orig_key}' in OCR text: '{txt}'")
                        matched_keys.add(orig_key)
                        ocr_key_hit.append({
                            "key": orig_key,
                            "text": txt,
                            "bbox": rec_polys[txt_idx],
                            "ocr_result_index": ocr_idx,
                            "text_index": txt_idx
                        })
                        break

    logging.debug(f"Key search complete: {len(matched_keys)} unique key(s) matched.")
    return matched_keys, ocr_key_hit


def refine_by_key_hits(matched, not_matched, hit_keys):
    logging.info(f"Refining by {len(hit_keys)} OCR-detected keys...")
    key_matched = {}
    key_not_matched = not_matched.copy()

    for attr_name, attr_obj in matched.items():
        if attr_name in hit_keys:
            key_matched[attr_name] = attr_obj
        else:
            new_obj = attr_obj.copy()
            new_obj["values"] = {v: False for v in attr_obj.get("values", [])}
            key_not_matched[attr_name] = new_obj

    logging.debug(f"After key refinement: {len(key_matched)} kept, {len(key_not_matched)} moved to not-matched.")
    return key_matched, key_not_matched


def refine_by_value_hits(key_matched, key_not_matched, big_text, schema):
    logging.info("Checking value presence in OCR text...")
    value_matched = {}
    value_not_matched = key_not_matched.copy()

    for key in key_matched:
        attr = get_attribute_info_by_key(key, schema)
        values = attr.get("values", [])
        hits = set(find_hits(big_text, values))

        value_map = {
            v: str(v).lower() in hits
            for v in values
        }

        new_obj = attr.copy()
        new_obj["values"] = value_map

        if any(value_map.values()):
            value_matched[key] = new_obj
        else:
            value_not_matched[key] = new_obj

    logging.debug(f"Value refinement: {len(value_matched)} attributes have matching values.")
    return value_matched, value_not_matched

def search_regex_in_text(regex_string: str, big_text: str):
    """
    Search for all matches of a regex string in a given text.

    Args:
        regex_string (str): The regex pattern to search for.
        big_text (str): The text to search within.

    Returns:
        list: A list of all matches found.
    """
    if not regex_string:
        return []

    try:
        pattern = re.compile(regex_string)
        matches = pattern.findall(big_text)
        return matches
    except re.error as e:
        print(f"Invalid regex: {e}")
        return []

def refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text, regex_withkey):
    final_value_matched = {}
    final_value_not_matched = value_not_matched.copy()

    for attr_name, attr_obj in value_matched.items():
        key = attr_obj.get("norm_key") or attr_obj.get("original_key")
        values = attr_obj.get("values", {})
        new_values = {}
        any_value_hit = False

        # Step 1: Check existing key-value pair hits
        for value, is_hit in values.items():
            if is_hit and matches_key_value_pair(big_text, key, value):
                new_values[value] = True
                any_value_hit = True
                print(f"✅ Matched key-value pair: {key} -> {value}")
            else:
                new_values[value] = False

        # Step 2: Additional regex-based matching from regex_withkey
        regex_info = regex_withkey.get(key, {})
        regex_pattern = regex_info.get("pair_regex")
        if regex_pattern:
            matches_from_regex = search_regex_in_text(regex_pattern, big_text)
            for match in matches_from_regex:
                # If the matched value exists in values, mark it True
                if match in new_values:
                    new_values[match] = True
                    any_value_hit = True
                    print(f"🔹 Regex matched key-value: {key} -> {match}")

        # Save the updated values back to the attribute
        new_attr = attr_obj.copy()
        new_attr["values"] = new_values

        if any_value_hit:
            final_value_matched[attr_name] = new_attr
        else:
            final_value_not_matched[attr_name] = new_attr

    return final_value_matched, final_value_not_matched

# def refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text):
#     logging.info("Validating key-value co-occurrence in text...")
#     final_value_matched = {}
#     final_value_not_matched = value_not_matched.copy()

#     for attr_name, attr_obj in value_matched.items():
#         key = attr_obj.get("norm_key") or attr_obj.get("original_key")
#         values = attr_obj.get("values", {})
#         new_values = {}
#         any_value_hit = False

#         for value, is_hit in values.items():
#             if is_hit and matches_key_value_pair(big_text, key, value):
#                 new_values[value] = True
#                 any_value_hit = True
#                 logging.info(f"✅ Matched key-value pair: {key} -> {value}")
#             else:
#                 new_values[value] = False

#         new_attr = attr_obj.copy()
#         new_attr["values"] = new_values

#         if any_value_hit:
#             final_value_matched[attr_name] = new_attr
#         else:
#             final_value_not_matched[attr_name] = new_attr

#     logging.info(f"Key-value validation complete: {len(final_value_matched)} attributes confirmed.")
#     return final_value_matched, final_value_not_matched
