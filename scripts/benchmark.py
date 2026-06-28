import sys
import os
import re
import json
import time
import argparse
import copy
import subprocess
import shutil
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor
from src.pipeline import MedicalOCRPipeline

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "pipeline_config.yaml")

# Levenshtein distance dynamic programming implementation
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def clean_for_eval(text: str) -> str:
    """
    Cleans structural markup, tags, punctuation, and normalizes spacing.
    Sorts words to bypass layout/reading order differences.
    """
    text = text.lower()
    # Remove tags like <s_ocr>, </s_ocr>, </s>, etc.
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove keys like doctor_name:, clinic_name:, etc.
    text = re.sub(r"[a-z_]+:", " ", text)
    # Remove bullet markers
    text = re.sub(r"\s-\s", " ", text)
    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Tokenize, sort, and join
    words = text.split()
    words.sort()
    return " ".join(words)

def calculate_cer(ref: str, hyp: str) -> float:
    ref_clean = clean_for_eval(ref)
    hyp_clean = clean_for_eval(hyp)
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = levenshtein_distance(ref_clean, hyp_clean)
    return dist / max(1, len(ref_clean))

def load_yaml_config(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def check_gpu_driver_support() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "nvidia_smi_available": False,
        "nvidia_driver_version": None,
        "nvidia_cuda_version": None,
        "onnxruntime_gpu": None,
        "bitsandbytes_importable": None,
    }

    info["bitsandbytes_importable"] = True
    try:
        import bitsandbytes as bnb
    except Exception:
        info["bitsandbytes_importable"] = False

    try:
        import onnxruntime as ort
        info["onnxruntime_gpu"] = ort.get_device() == "GPU"
    except Exception:
        info["onnxruntime_gpu"] = False

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        info["nvidia_smi_available"] = True
        try:
            result = subprocess.run([nvidia_smi, "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                values = result.stdout.strip().split(",")
                info["nvidia_driver_version"] = values[0].strip() if values else None
                info["nvidia_cuda_version"] = values[1].strip() if len(values) > 1 else None
        except Exception:
            pass

    return info


def configure_pipeline_for_variant(base_config: dict, model_variant: str, mode: str, device: str) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault("hardware", {})
    config["hardware"]["device"] = device
    config["hardware"]["parallel_ocr"] = mode == "parallel"
    config["hardware"]["parallel_ocr_workers"] = 2

    if model_variant == "trocr-small-handwritten":
        config.setdefault("models", {}).setdefault("handwriting_ocr_model", {})["name"] = "microsoft/trocr-small-handwritten"
    elif model_variant == "trocr-base-handwritten":
        config.setdefault("models", {}).setdefault("handwriting_ocr_model", {})["name"] = "microsoft/trocr-base-handwritten"
    elif model_variant == "default":
        pass
    else:
        # allow explicit variant strings of the form printed=...,handwritten=...
        if "," in model_variant:
            pairs = [p.strip() for p in model_variant.split(",") if p.strip()]
            for p in pairs:
                if "=" in p:
                    k, v = p.split("=", 1)
                    key = k.strip()
                    value = v.strip()
                    if key == "printed":
                        config.setdefault("models", {}).setdefault("printed_ocr_model", {})["rec_name"] = value
                    elif key == "handwritten":
                        config.setdefault("models", {}).setdefault("handwriting_ocr_model", {})["name"] = value
    return config


def build_model_variants(variant_string: str) -> list[str]:
    if not variant_string:
        return ["default"]
    return [v.strip() for v in variant_string.split(",") if v.strip()]


def benchmark_images(pipeline: MedicalOCRPipeline, image_paths: list[str]) -> dict:
    latencies = []
    vrams = []
    oom_count = 0
    stage_timings = {}

    for idx, img_path in enumerate(image_paths):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        img = cv2.imread(img_path)
        if img is None:
            continue

        start_time = time.time()
        try:
            res = pipeline.process_image(img, skip_summarization=True)
            elapsed = time.time() - start_time
            latencies.append(elapsed)

            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            vrams.append(peak_vram)

            for stage, t_val in res["timings"].items():
                if isinstance(t_val, (int, float)):
                    stage_timings.setdefault(stage, []).append(t_val)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

    results = {
        "samples": len(latencies),
        "average_latency": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p90_latency": float(np.percentile(latencies, 90)) if latencies else 0.0,
        "min_latency": float(np.min(latencies)) if latencies else 0.0,
        "max_latency": float(np.max(latencies)) if latencies else 0.0,
        "throughput_images_per_sec": float(len(latencies) / sum(latencies)) if latencies and sum(latencies) > 0 else 0.0,
        "peak_vram_mb": float(np.max(vrams)) if vrams else 0.0,
        "average_vram_mb": float(np.mean(vrams)) if vrams else 0.0,
        "oom_count": oom_count,
        "oom_rate_pct": (oom_count / len(image_paths) * 100) if image_paths else 0.0,
        "stage_averages_sec": {k: float(np.mean(v)) for k, v in stage_timings.items()},
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Medical OCR Pipeline Benchmark Runner")
    parser.add_argument("--dataset_dir", type=str, default="medical-prescription-dataset", help="Path to prescription dataset")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples for accuracy evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_json", type=str, default="benchmark_report.json", help="Path to save output JSON report")
    parser.add_argument("--mode", type=str, choices=["sequential", "parallel", "both"], default="both", help="OCR execution mode")
    parser.add_argument("--model_variants", type=str, default="default", help="Comma-separated list of model variants to compare")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to pipeline YAML config")
    args = parser.parse_args()

    base_config = load_yaml_config(args.config) if os.path.exists(args.config) else {}
    driver_info = check_gpu_driver_support()
    print("GPU/driver status:", json.dumps(driver_info, indent=2))

    real_world_images = []
    kastoor_dir = "Patient_Kastoor"
    if os.path.exists(kastoor_dir):
        for f in os.listdir(kastoor_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                real_world_images.append(os.path.join(kastoor_dir, f))
    whatsapp_dir = "WhatsApp.Unknown.2026-04-27.at.12.10.10"
    if os.path.exists(whatsapp_dir):
        for f in sorted(os.listdir(whatsapp_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                real_world_images.append(os.path.join(whatsapp_dir, f))

    variant_names = build_model_variants(args.model_variants)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "driver_info": driver_info,
        "runs": []
    }

    for model_variant in variant_names:
        for mode in ([args.mode] if args.mode != "both" else ["sequential", "parallel"]):
            config = configure_pipeline_for_variant(base_config, model_variant, mode, args.device)
            print(f"\n--- Running benchmark for variant '{model_variant}' in {mode} mode ---")
            pipeline = MedicalOCRPipeline(config=config)
            run_result = benchmark_images(pipeline, real_world_images)
            run_result.update({
                "variant": model_variant,
                "mode": mode,
                "image_count": len(real_world_images),
                "config_source": args.config,
            })
            results["runs"].append(run_result)

    # Accuracy benchmark: no model variant looping here, use default
    test_img_dir = os.path.join(args.dataset_dir, "test", "images")
    test_anno_dir = os.path.join(args.dataset_dir, "test", "annotations")
    cers = []
    accuracy_samples_processed = 0
    if os.path.exists(test_img_dir) and os.path.exists(test_anno_dir):
        print(f"\n--- Accuracy Benchmarking on {args.samples} Prescription Test Images ---")
        test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])[:args.samples]
        pipeline = MedicalOCRPipeline(config=configure_pipeline_for_variant(base_config, "default", "sequential", args.device))
        for img_name in test_images:
            base_name = os.path.splitext(img_name)[0]
            anno_name = f"{base_name}.json"
            img_path = os.path.join(test_img_dir, img_name)
            anno_path = os.path.join(test_anno_dir, anno_name)
            if not os.path.exists(anno_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            with open(anno_path, "r", encoding="utf-8") as f:
                anno_data = json.load(f)
            ref_text = anno_data.get("ground_truth", "")
            try:
                res = pipeline.process_image(img, skip_summarization=True)
                hyp_text = res.get("corrected_ocr_text", "")
                cer = calculate_cer(ref_text, hyp_text)
                cers.append(cer)
                accuracy_samples_processed += 1
                print(f"  {img_name} | CER: {cer * 100:.2f}%")
            except Exception as e:
                print(f"  {img_name} failed: {e}")
    avg_cer = float(np.mean(cers)) if cers else 0.0
    results["accuracy_stats"] = {
        "samples": accuracy_samples_processed,
        "average_cer": avg_cer,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved benchmark report: {args.output_json}")

    print("\n## SUMMARY")
    for run in results["runs"]:
        print(f"Variant={run['variant']} Mode={run['mode']} Images={run['image_count']} Throughput={run['throughput_images_per_sec']:.2f} ips p50={run['p50_latency']:.2f}s p90={run['p90_latency']:.2f}s PeakVRAM={run['peak_vram_mb']:.1f}MB OOM={run['oom_count']}")
    print(f"Accuracy (default variant): avg CER = {avg_cer * 100:.2f}% on {accuracy_samples_processed} samples")

if __name__ == "__main__":
    main()
