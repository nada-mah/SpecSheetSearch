import argparse
import glob
import os
import shutil
from process_lighting_spec_sheet import process_lighting_spec_sheet
from model_loader import _ocr_instance

def main():
    parser = argparse.ArgumentParser(description="Extract structured lighting specs from PDF spec sheets.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (handled internally by model loader)")
    parser.add_argument("--input", required=True, type=str, help="Path to folder containing input PDF files")
    parser.add_argument("--schema", required=True, type=str, help="Path to schema JSON file")

    args = parser.parse_args()

    input_pdf_folder = args.input
    schema_path = args.schema

    if not os.path.isdir(input_pdf_folder):
        raise ValueError(f"Input folder does not exist: {input_pdf_folder}")
    if not os.path.isfile(schema_path):
        raise ValueError(f"Schema file not found: {schema_path}")

    # Find all PDFs (case-insensitive)
    pdf_paths = (
        glob.glob(os.path.join(input_pdf_folder, "*.pdf")) +
        glob.glob(os.path.join(input_pdf_folder, "*.PDF"))
    )

    if not pdf_paths:
        print(f"⚠️ No PDF files found in {input_pdf_folder}")
        return

    # Define output structure
    output_dir = "final_result"
    success_dir = os.path.join(output_dir, "success_found")
    not_found_dir = os.path.join(output_dir, "not_found")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(not_found_dir, exist_ok=True)

    print(f"📄 Found {len(pdf_paths)} PDF(s) to process.\n")

    # Process each PDF
    for pdf_path in pdf_paths:
        try:
            print(f"\n--- Processing: {os.path.basename(pdf_path)} ---")
            is_hit = process_lighting_spec_sheet(
                pdf_path,
                schema_path,
                _ocr_instance,
                output_dir=output_dir,
                use_gpu=args.gpu
            )

            filename = os.path.basename(pdf_path)
            if is_hit:
                dest = os.path.join(success_dir, filename)
                print(f"✅ Success: moving {filename} to success folder")
            else:
                dest = os.path.join(not_found_dir, filename)
                print(f"❌ No match: moving {filename} to not_found folder")

            # Move original PDF
            shutil.move(pdf_path, dest)

        except Exception as e:
            print(f"⚠️ Error processing {pdf_path}: {e}")
            # Optionally move to an error folder (not implemented here)

    print("\n✨ All done!")

if __name__ == "__main__":
    main()