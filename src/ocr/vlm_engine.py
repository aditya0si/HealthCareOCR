import os
import time
import torch
import json
import logging
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

logger = logging.getLogger(__name__)

class VLMEngine:
    def __init__(self, config: dict = None):
        """
        Track A: Unified Vision Language Model OCR and Clinical Summarizer.
        """
        self.config = config if config is not None else {}
        self.device = self.config.get("device", "cuda")
        self.model_name = self.config.get("models", {}).get("vlm", {}).get("name", "Qwen/Qwen2-VL-2B-Instruct")
        self.min_pixels = self.config.get("models", {}).get("vlm", {}).get("min_pixels", 256 * 256)
        self.max_pixels = self.config.get("models", {}).get("vlm", {}).get("max_pixels", 768 * 768)
        
        self.processor = None
        self.model = None

    def load_model(self):
        """
        Lazy-loads the processor and model in 4-bit NF4 to minimize VRAM footprint.
        """
        if self.model is not None:
            return

        print(f"[VLM] Loading Qwen2-VL model: {self.model_name} in 4-bit...")
        start_time = time.time()

        # Initialize processor with pixel bounds
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

        # Configure bitsandbytes 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            attn_implementation="sdpa"
        )
        self.model.eval()
        print(f"[VLM] Model loaded successfully in {time.time() - start_time:.2f}s")

    def recognize_and_summarize(self, image: np.ndarray) -> dict:
        """
        Performs end-to-end OCR and Clinical Summarization in a single VLM pass.
        Returns a dictionary conforming to the structured clinical summary schema.
        """
        self.load_model()

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        else:
            pil_img = image

        # Construct prompt enforcing the JSON structure
        json_schema = {
            "patient_name": "string or null",
            "age_sex": "string or null",
            "document_type": "lab_report | prescription | discharge_summary | other",
            "date": "string (e.g. YYYY-MM-DD or raw string) or null",
            "hospital": "string or null",
            "doctor": "string or null",
            "key_findings": ["list of strings"],
            "medications": [{"drug": "string", "dosage": "string or null", "frequency": "string or null"}],
            "diagnoses": ["list of strings"],
            "abnormal_values": [{"test": "string", "value": "string", "reference": "string or null", "status": "high | low | normal"}],
            "summary": "a 2-3 sentence clinical summary"
        }

        schema_str = json.dumps(json_schema, indent=2)
        prompt = (
            "You are an expert clinical transcription and information extraction assistant. "
            "Analyze the medical report image and generate two outputs in sequence:\n\n"
            "1. Output a JSON object representing the clinical summary matching this schema:\n"
            f"{schema_str}\n\n"
            "2. Output the delimiter text exactly on a new line: '---TRANSCRIPTION---'\n"
            "3. Under the delimiter, perform a complete, line-by-line transcription of all printed and handwritten text found in the image. Do not skip any sections.\n\n"
            "Do not wrap the JSON in markdown code fences like ```json. "
            "Do not include any other introductory or concluding text."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=pil_img,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        print("[VLM] Running unified inference pass...")
        start_gen = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                use_cache=True
            )

        # Trim prompt inputs from outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        gen_time = time.time() - start_gen
        print(f"[VLM] Inference finished in {gen_time:.2f}s")

        # Parse JSON and transcription from output
        raw_text = ""
        json_part = output_text
        if "---TRANSCRIPTION---" in output_text:
            parts = output_text.split("---TRANSCRIPTION---", 1)
            json_part = parts[0].strip()
            raw_text = parts[1].strip()
        else:
            # Fallback split if delimiter is slightly changed
            for delim in ["---transcription---", "transcription:", "TRANSCRIPTION:"]:
                if delim in output_text.lower():
                    idx = output_text.lower().find(delim)
                    json_part = output_text[:idx].strip()
                    raw_text = output_text[idx + len(delim):].strip()
                    break

        # Clean markdown wrappers from output
        cleaned_json_part = json_part.strip()
        if cleaned_json_part.startswith("```"):
            lines = cleaned_json_part.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_json_part = "\n".join(lines).strip()

        # Parse JSON output
        try:
            summary_data = json.loads(cleaned_json_part)
            # Ensure it has raw_ocr_text
            summary_data["raw_ocr_text"] = raw_text if raw_text else output_text
        except Exception as e:
            logger.exception("Failed to parse VLM JSON output; wrapping raw output as backup: %s", e)
            print(f"[VLM] JSON parsing failed: {e}. Raw text starts with: {cleaned_json_part[:150]}")
            # Fallback structure
            summary_data = {
                "patient_name": None,
                "age_sex": None,
                "document_type": "other",
                "date": None,
                "hospital": None,
                "doctor": None,
                "key_findings": [],
                "medications": [],
                "diagnoses": [],
                "abnormal_values": [],
                "raw_ocr_text": output_text,
                "summary": "Failed to parse structured summary. Raw text is captured in transcription tab."
            }

        # Add performance metrics to response
        metrics = {
            "vlm_load_time": 0.0, # cached
            "vlm_inference_time": gen_time
        }
        
        return summary_data, metrics

    def unload(self):
        """
        Moves model to CPU and clears GPU cache to free VRAM when not in use.
        """
        if self.model is not None:
            print("[VLM] Swapping model to CPU...")
            try:
                self.model.to("cpu")
            except Exception as e:
                print(f"[VLM] Failed to move model to CPU: {e}")
            torch.cuda.empty_cache()
            print("[VLM] Swapped successfully.")
