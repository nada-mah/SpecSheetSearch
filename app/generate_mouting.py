import logging
import re
import json
from  input_handler import load_attribute_schema
from  model_loader import get_llm_instance

def build_mounting_prompt(product_type_set):
    prompt = f'''
You are a helpful assistant. You are a lighting product domain expert focused on lighting manufacturers and their products.
Your knowledge domain is defined by information typically found in:
Manufacturer spec sheets
Product cut sheets
Digital spec sheets / digital cutsheets
Lighting Exchange product listings
You understand lighting products across all categories, including architectural, commercial, residential, industrial, outdoor, emergency, and lighting controls.
You are given a list of lighting product types.
{product_type_set}

For each product type, identify the most typical mounting description based on common lighting manufacturer specifications and industry usage.

If a product type commonly uses more than one mounting description, return a list of mounting words ordered from most specific to most general.

Mounting descriptions should use common lighting-industry terms.

**Examples of typical mounting terms include, but are not limited to:**

- **Primary mounting styles:**
  recessed, surface, suspended, pendant, wall, pole, ceiling, ingrade, ground

- **Mounting methods and supports:**
  cable, stem, chain, truss, canopy, magnetic, junction

- **Specialized or application-specific mountings:**
  portable, underwater, submersible, highmast, stake, bracket, arm

Use these examples as guidance rather than a strict list.
Prefer widely recognized industry terminology over generic physical descriptions.
Respond with JSON only.

Each key must be the original product type.
Each value must be either:

- a single mounting word

Do not include explanations, comments, markdown, or any text outside the JSON object.
Return ONLY valid JSON.
Do not include explanations, markdown, or comments.
Use double quotes for all keys and string values.
'''
    return prompt


def generate_llm_response(prompt, use_gpu=False):
    llm = get_llm_instance(use_gpu)
    response =  llm.create_chat_completion(
    messages=[
        # {
            # "role": "system",
            # # "content": "You are a helpful assistant that outputs in JSON format. Output only valid JSON. Ensure all keys are quoted, no duplicate keys exist, and arrays contain uniform types.",
            # # "content": "You are a JSON-specialized assistant with strict formatting requirements. Your output MUST: 1) Be valid JSON (validate with JSONLint) 2) Quote all keys with double quotes 3) No duplicate keys 4) Arrays with uniform types only 5) Use proper escaping for special characters 6) Match the exact schema shown in the example 7) Never include markdown formatting. Prioritize format validity over content completeness. Output only the JSON object without additional text.",
            # "content": "You are a JSON-specialized assistant with strict formatting requirements. first think about your task. Your output MUST: 1) Be valid JSON (validate with JSONLint) 2) Quote all keys with double quotes 3) No duplicate keys 4) Arrays with uniform types only 5) Use proper escaping for special characters 6) Match the exact schema shown in the example. Always Reply in this format: 'Thoughts': { brief reasoning:}, 'JSON': {}",
            {
            "role": "system",
            "content": """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are a JSON-specialized assistant with strict formatting rules. Do not output markdown or natural language. Do not overthink or explain your steps. Think carefully and analyze the input thoroughly before responding. Your output MUST:

            1. Be valid JSON (passes JSONLint)
            2. Quote all keys with double quotes
            3. Contain no duplicate keys
            4. Use arrays with uniform types
            5. Use proper escaping for special characters
            7. Never include markdown, commentary, or explanations

            Output only the JSON object. Nothing else.
            \no_think
            """
            },
        {"role": "user", "content": prompt},
    ],
    temperature=0.9,
    max_tokens=0.9,
    top_p=10,
    )
    content = response["choices"][0]["message"]['content']
    return content

def load_schema_and_derive_product_types(schema_path):
    logging.info(f"Loading schema from {schema_path}...")
    schema = load_attribute_schema(schema_path)
    product_type_set = set()
    for key in schema:
        product_type_set.update(schema[key]["product_types"])
    logging.info(f"Derived {len(product_type_set)} unique product types.")
    return schema, product_type_set


def fix_and_load_json(response_text):
    logging.info("Attempting to parse and sanitize JSON response...")
    response_text = response_text.strip()

    # Extract JSON block if wrapped in text
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        response_text = match.group(0)

    # Remove trailing commas ONLY
    response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)

    try:
        parsed = json.loads(response_text)
        logging.debug("JSON successfully parsed.")
        return parsed
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON: {e}")
        raise ValueError(f"Invalid JSON: {e}")


def get_valid_json(prompt, initial_response=None, max_retries=3, use_gpu=False):
    """
    Validates an initial LLM response and regenerates only if necessary.
    """
    if initial_response is not None:
        try:
            return fix_and_load_json(initial_response)
        except ValueError:
            logging.info("Initial response is invalid JSON; regenerating...")

    for attempt in range(max_retries):
        logging.info(f"LLM JSON generation attempt {attempt + 1}/{max_retries}")
        response = generate_llm_response(prompt, use_gpu)
        try:
            return fix_and_load_json(response)
        except ValueError:
            logging.warning(f"Attempt {attempt + 1} failed: invalid JSON. Retrying...")

    raise RuntimeError("Failed to get valid JSON after multiple attempts")

