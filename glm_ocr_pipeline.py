"""
GLM-OCR Python Pipeline
------------------------
Batch OCR for images and PDFs using zai-org/GLM-OCR.

Install:
    pip install git+https://github.com/huggingface/transformers.git
    pip install torch pillow pdf2image accelerate

PDF support needs Poppler installed on your system (see README notes).

Usage:
    python glm_ocr_pipeline.py --input ./docs --output ./results --mode text

    Or import GLMOcr in your own code:
        from glm_ocr_pipeline import GLMOcr
        ocr = GLMOcr()
        text = ocr.run_on_image("invoice.png", mode="text")
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "zai-org/GLM-OCR"

PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class GLMOcr:
    """Thin wrapper around GLM-OCR for repeated inference calls."""

    def __init__(self, model_path: str = MODEL_PATH, max_new_tokens: int = 8192, poppler_path: str = None):
        self.max_new_tokens = max_new_tokens
        self.poppler_path = poppler_path
        print(f"Initializing GLMOcr with model: {model_path} ...", flush=True)
        print(f"Loading processor ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print(f"Loading model weights ... (first run will download ~2.6GB)", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        print(f"Model loaded successfully on device: {self.model.device}", flush=True)

    def _generate(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output_text.strip()

    def run_on_image(self, image_path: str, mode: str = "text") -> str:
        """mode: 'text' | 'formula' | 'table'"""
        if mode not in PROMPTS:
            raise ValueError(f"mode must be one of {list(PROMPTS)}")
        image = Image.open(image_path).convert("RGB")
        return self._generate(image, PROMPTS[mode])

    def run_extraction(self, image_path: str, schema: dict, language_hint: str = "") -> dict:
        """
        Structured information extraction against a JSON schema.
        Returns parsed JSON (falls back to raw string under 'raw' key if parsing fails).
        """
        image = Image.open(image_path).convert("RGB")
        prompt = (
            f"{language_hint}\n" if language_hint else ""
        ) + "请按下列JSON格式输出图中信息:\n" + json.dumps(schema, ensure_ascii=False, indent=2)
        raw = self._generate(image, prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def run_on_pdf(self, pdf_path: str, mode: str = "text", dpi: int = 200, poppler_path: str = None) -> list[str]:
        """Converts each PDF page to an image and OCRs it. Returns list of page texts."""
        from pdf2image import convert_from_path
        from tqdm import tqdm

        if poppler_path is None:
            poppler_path = self.poppler_path

        print(f"  Converting PDF to images with Poppler (DPI={dpi}) ...", flush=True)
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        except Exception as e:
            print(f"  ERROR during PDF conversion: {e}", flush=True)
            raise
        print(f"  PDF converted successfully. Total pages to OCR: {len(pages)}", flush=True)

        results = []
        # Use tqdm progress bar for page-by-page progress
        for i, page_image in enumerate(tqdm(pages, desc="  OCRing Pages", unit="page"), start=1):
            print(f"  Processing page {i}/{len(pages)} ...", flush=True)
            text = self._generate(page_image, PROMPTS[mode])
            results.append(text)
        return results


def process_folder(input_dir: str, output_dir: str, mode: str, poppler_path: str = None, dpi: int = 200):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning folder: {input_dir}", flush=True)
    files = sorted(input_path.iterdir())
    supported_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() == ".pdf"]

    if not supported_files:
        print(f"No supported image or PDF files found in {input_dir}", flush=True)
        return

    print(f"Found {len(supported_files)} supported file(s). Starting OCR pipeline ...", flush=True)
    
    # Initialize GLMOcr engine
    ocr = GLMOcr(poppler_path=poppler_path)

    for idx, f in enumerate(supported_files, start=1):
        ext = f.suffix.lower()
        print(f"\n[{idx}/{len(supported_files)}] Processing: {f.name}", flush=True)
        try:
            if ext in IMAGE_EXTENSIONS:
                text = ocr.run_on_image(str(f), mode=mode)
                out_file = output_path / f"{f.stem}.md"
                out_file.write_text(text, encoding="utf-8")
                print(f"  -> Saved output to: {out_file}", flush=True)

            elif ext == ".pdf":
                page_texts = ocr.run_on_pdf(str(f), mode=mode, dpi=dpi)
                out_file = output_path / f"{f.stem}.md"
                out_file.write_text(
                    "\n\n---\n\n".join(page_texts), encoding="utf-8"
                )
                print(f"  -> Saved output to: {out_file} ({len(page_texts)} pages)", flush=True)
        except Exception as e:
            print(f"  ERROR: Failed to process {f.name}: {e}", flush=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch OCR with GLM-OCR")
    parser.add_argument("--input", required=True, help="Folder of images/PDFs to OCR")
    parser.add_argument("--output", required=True, help="Folder to write .md results to")
    parser.add_argument(
        "--mode", default="text", choices=list(PROMPTS.keys()),
        help="text | formula | table"
    )
    parser.add_argument(
        "--poppler-path", default=None, help="Path to poppler bin directory"
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="DPI for PDF page conversion (lower is faster on CPU, e.g. 100-150)"
    )
    args = parser.parse_args()

    process_folder(args.input, args.output, args.mode, args.poppler_path, args.dpi)
