import logging
import itertools

def generate_ocr_variants(term):
    """
    Generates common OCR-friendly variants for a string/number.
    Example: '109' -> ['109', '1o9', 'l09', 'IO9']
    """
    term = str(term)
    original_lower = term.lower()
    variants = {original_lower}

    # Map common OCR confusions
    ocr_map = {
        '0': ['0', 'o', 'O'],
        '1': ['1', 'l', 'I', 'i'],
        'i': ['i', 'l', '1', 'I'],
        'l': ['l', 'i', '1', 'I'],
        '5': ['5', 'S', 's'],
        '8': ['8', 'B'],
        '9': ['9', 'g', 'q'],
        '2': ['2', 'Z', 'z'],
        '6': ['6', 'G', 'b']
    }

    # Build character options using lowercase base
    chars_options = []
    for c in term:
        c_lower = c.lower()
        replacements = ocr_map.get(c_lower, [c_lower])
        # Ensure all choices are lowercase to avoid case-related duplication
        chars_options.append([r.lower() for r in replacements])

    # Generate all combinations
    total_combos = 1
    for options in chars_options:
        total_combos *= len(options)
    
    # Optional: warn if combinatorial explosion (e.g., >10k variants)
    if total_combos > 10000:
        logging.warning(f"Large number of OCR variants generated for '{term}' ({total_combos} combinations). Performance may be affected.")

    for combo in itertools.product(*chars_options):
        variants.add(''.join(combo))

    variant_list = sorted(variants)
    logging.debug(f"Generated {len(variant_list)} OCR variant(s) for term '{term}': {variant_list}")
    return variant_list


