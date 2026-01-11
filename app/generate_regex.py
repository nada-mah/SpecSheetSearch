def build_regex_prompt(input):
  Regex_prompt = f'''
You are given a JSON object where each key has a list of "values" and "Expected Output Formatting" rules.

## TASK:
For EACH top-level object, generate  "pair_regex".

## CONDITIONAL LOGIC (CRITICAL):
- **NULL RULE:** If the "Expected Output Formatting" indicates the value must "exactly match" the predefined list (even if case/spacing differs), set BOTH regex fields to `null`.
- **REGEX RULE:** Only generate a regex if the formatting allows for **Numeric measurements** (e.g., 4in, 2x2), **Percentages** (e.g., 30%), or **Variants/Similar text** not explicitly listed in the "values" array.
- **NO REDUNDANCY:** Do not generate a regex that is simply a long list of the "values" joined by OR (|) operators. If no dynamic pattern (like digits or wildcards) is needed, return `null`.
- ALSO if Expected Output Formatting = [Must exactly match one of the predefined option values with true/false statement] or smilar then pair_regex WILL BE null

## DEFINITIONS:
- "pair_regex": Matches the object name followed by a separator (colon/space/dash).

## OUTPUT RULES:
- Output MUST be valid JSON only.
- Output MUST contain ONLY the original top-level keys.
- Use `(?i)` for case-insensitivity.

## OUTPUT FORMAT:
{{
  "<object_name>": {{
    "pair_regex": "..." or null
  }}
}}

## INPUT:
{input}
/no_think
'''
  return Regex_prompt


import re
from collections import defaultdict

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\(.*?\)', '', text)   # remove parentheticals
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def prefix_key(text, prefix_len=4):
    """
    Returns the first `prefix_len` tokens as a grouping key.
    """
    tokens = text.split()
    return " ".join(tokens[:prefix_len])
from collections import defaultdict
from difflib import SequenceMatcher

def sentence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def group_schema_by_sentence_closeness(normalized_schema, threshold=0.5):
    """
    Groups attributes by how close their formatting sentences are.
    """

    items = []
    for name, details in normalized_schema.items():
        instr = details.get("Expected Output Formatting", "")
        items.append({
            "name": name,
            "details": details,
            "text": normalize_text(instr)
        })

    groups = []
    visited = set()

    for i, base in enumerate(items):
        if i in visited:
            continue

        group = {base["name"]: base["details"]}
        visited.add(i)

        for j in range(i + 1, len(items)):
            if j in visited:
                continue

            sim = sentence_similarity(base["text"], items[j]["text"])
            if sim >= threshold:
                group[items[j]["name"]] = items[j]["details"]
                visited.add(j)

        groups.append(group)

    return groups

    return list(groups.values())



def clean_guidance(guidance_list):
    # Keys we want to remove
    keys_to_remove = {'data_type', 'product_types'}

    return [
        {
            key: {
                k: v for k, v in value.items() if k not in keys_to_remove
            }
            for key, value in item.items()
        }
        for item in guidance_list
    ]

# Usage:
# cleaned_guidance = clean_guidance(guidance)