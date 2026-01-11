# pdfs = ['/content/Metalux-14GRLED-1-x-4-LED-Troffer-2000-to-4300-Lumens-Package-spec-sheet.pdf']

import glob
import os
import shutil
from app.process_lighting_spec_sheet import process_lighting_spec_sheet
from app.model_loader import _ocr_instance, tokenizer, modelq
input_pdf_folder = "/content"
pdf_paths = glob.glob(os.path.join(input_pdf_folder, "*.pdf")) + \
            glob.glob(os.path.join(input_pdf_folder, "*.PDF"))

schema_path = "/content/testdata.json"

# Define folders
output_dir = "final_result"
success_dir = f"{output_dir}/success_found"
not_found_dir = f"{output_dir}/not_found"

# Create all needed directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(success_dir, exist_ok=True)
os.makedirs(not_found_dir, exist_ok=True)


# Process each PDF
for pdf_path in pdf_paths:
    try:
        is_hit = process_lighting_spec_sheet(
            pdf_path,
            schema_path,
            _ocr_instance,
            tokenizer,
            modelq,
            output_dir=output_dir
        )

        filename = os.path.basename(pdf_path)
        if is_hit:
            dest = os.path.join(success_dir, filename)
            print(f"✅ Success: moving {filename} to {success_dir}")
        else:
            dest = os.path.join(not_found_dir, filename)
            print(f"❌ No match: moving {filename} to {not_found_dir}")

        shutil.move(pdf_path, dest)

    except Exception as e:
        print(f"⚠️ Error processing {pdf_path}: {e}")