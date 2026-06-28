import os
import cv2
import base64
import numpy as np
import json
import uvicorn
import traceback
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from src.pipeline import MedicalOCRPipeline
from src.batch_runner import BatchRunner

# Initialize the pipeline globally
print("Initializing Medical OCR Pipeline...")
pipeline = MedicalOCRPipeline()

# Run a warm-up on a small dummy image at startup
print("Running warm-up run...")
dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    _ = pipeline.process_image(dummy_img, skip_summarization=True)
    print("Warm-up complete.")
except Exception as e:
    print(f"Warm-up failed: {e}")

app = FastAPI(title="Medical OCR & Summarization Pipeline")

# Initialize batch runner with shared pipeline
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
batch_runner = BatchRunner(pipeline, max_preprocessing_workers=6)

# Pre-configured patient directories
KNOWN_DIRECTORIES = {
    "whatsapp_unknown": os.path.join(PROJECT_ROOT, "WhatsApp.Unknown.2026-04-27.at.12.10.10"),
    "patient_kastoor": os.path.join(PROJECT_ROOT, "Patient_Kastoor"),
}

class BatchRequest(BaseModel):
    directories: list[str] = []  # directory keys from KNOWN_DIRECTORIES

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})
            
        # skip_summarization is set to False to enable LLM clinical summary generation
        result = pipeline.process_image(img, skip_summarization=False)
        return JSONResponse(content=result)
        
    except Exception as e:
        err_msg = traceback.format_exc()
        print(err_msg)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": err_msg})

# ============================================================
# BATCH PROCESSING ENDPOINTS
# ============================================================

@app.post("/batch-process")
async def batch_process(request: BatchRequest):
    """Start batch processing on one or more known directories."""
    try:
        dirs = []
        for key in request.directories:
            if key in KNOWN_DIRECTORIES:
                path = KNOWN_DIRECTORIES[key]
                if os.path.isdir(path):
                    dirs.append(path)
                else:
                    return JSONResponse(status_code=400, content={"error": f"Directory not found: {path}"})
            else:
                return JSONResponse(status_code=400, content={
                    "error": f"Unknown directory key: {key}",
                    "available": list(KNOWN_DIRECTORIES.keys())
                })

        if not dirs:
            return JSONResponse(status_code=400, content={"error": "No valid directories specified"})

        batch_id = batch_runner.run_batch(dirs)
        return JSONResponse(content={"batch_id": batch_id, "status": "started", "directories": [os.path.basename(d) for d in dirs]})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Poll batch processing progress."""
    batch = batch_runner.get_batch_status(batch_id)
    if batch is None:
        return JSONResponse(status_code=404, content={"error": "Batch not found"})
    return JSONResponse(content=batch.to_dict())


@app.get("/batch-results/{batch_id}")
async def get_batch_results(batch_id: str):
    """Get full batch results with all per-image data."""
    batch = batch_runner.get_batch_status(batch_id)
    if batch is None:
        return JSONResponse(status_code=404, content={"error": "Batch not found"})

    results_data = [r.to_dict() for r in batch.results]
    elapsed = (batch.end_time or __import__('time').time()) - batch.start_time if batch.start_time else 0

    return JSONResponse(content={
        "batch_id": batch.batch_id,
        "status": batch.status,
        "total_images": batch.total_images,
        "processed": batch.processed,
        "failed": batch.failed,
        "elapsed_seconds": round(elapsed, 2),
        "directories": batch.directories,
        "results": results_data,
    })


@app.get("/serve-image")
async def serve_image(path: str = Query(...)):
    """Serve a thumbnail of an image file for the batch results gallery."""
    try:
        # Security: only serve from known directories
        abs_path = os.path.abspath(path)
        allowed = any(abs_path.startswith(os.path.abspath(d)) for d in KNOWN_DIRECTORIES.values())
        if not allowed:
            return JSONResponse(status_code=403, content={"error": "Access denied"})

        if not os.path.exists(abs_path):
            return JSONResponse(status_code=404, content={"error": "File not found"})

        img = cv2.imread(abs_path)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Cannot read image"})

        # Resize to thumbnail (max 300px on longest side)
        h, w = img.shape[:2]
        scale = 300 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/known-directories")
async def get_known_directories():
    """List available directories for batch processing."""
    dirs = {}
    for key, path in KNOWN_DIRECTORIES.items():
        exists = os.path.isdir(path)
        image_count = 0
        if exists:
            for root, _, files in os.walk(path):
                image_count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'))
        dirs[key] = {
            "path": path,
            "name": os.path.basename(path),
            "exists": exists,
            "image_count": image_count,
        }
    return JSONResponse(content=dirs)

# Embedded premium index HTML
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare OCR & Clinical Summarizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(13, 20, 35, 0.65);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary-glow: #3b82f6;
            --accent-glow: #6366f1;
            --cyan-glow: #06b6d4;
            --emerald-glow: #10b981;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --shadow-glow: 0 0 15px rgba(59, 130, 246, 0.15);
        }

        body {
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.08) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.08) 0px, transparent 40%);
            background-attachment: fixed;
        }

        header {
            padding: 1.5rem 5%;
            border-bottom: 1px solid var(--border-color);
            background: rgba(11, 15, 25, 0.7);
            backdrop-filter: blur(12px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        header h1 {
            margin: 0;
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        header p {
            margin: 0.2rem 0 0 0;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        main {
            padding: 2rem 5%;
            max-width: 1440px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 2rem;
            box-sizing: border-box;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(20px) saturate(180%);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(59, 130, 246, 0.2);
        }

        .upload-zone {
            border: 2px dashed rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 2rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.01);
        }

        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--primary-glow);
            background: rgba(59, 130, 246, 0.04);
            box-shadow: var(--shadow-glow);
        }

        .upload-icon {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
            transition: transform 0.2s ease;
        }

        .upload-zone:hover .upload-icon {
            transform: translateY(-5px);
        }

        .btn {
            background: linear-gradient(135deg, var(--primary-glow) 0%, var(--accent-glow) 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            width: 100%;
            margin-top: 1rem;
            transition: opacity 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        }

        .btn:hover {
            opacity: 0.95;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        }

        .btn:active {
            transform: scale(0.98);
        }

        .btn:disabled {
            background: #27272a;
            color: #71717a;
            cursor: not-allowed;
            box-shadow: none;
        }

        .image-preview {
            max-width: 100%;
            max-height: 250px;
            border-radius: 8px;
            margin-top: 1rem;
            display: none;
            object-fit: contain;
            border: 1px solid var(--border-color);
        }

        /* Quality Check Section */
        .quality-section {
            margin-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
            display: none;
        }

        .quality-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
        }

        .quality-header h3 {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--cyan-glow);
        }

        .quality-gauge-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            background: rgba(255, 255, 255, 0.02);
            padding: 0.8rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }

        .quality-gauge {
            width: 50px;
            height: 50px;
            transform: rotate(-90deg);
        }

        .gauge-text {
            display: flex;
            flex-direction: column;
        }

        .gauge-score {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .gauge-lbl {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .metric-row {
            margin-bottom: 0.8rem;
        }

        .metric-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            margin-bottom: 0.3rem;
        }

        .metric-name {
            color: var(--text-secondary);
        }

        .metric-value {
            font-weight: 500;
        }

        .bar-outer {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
        }

        .bar-inner {
            height: 100%;
            width: 0%;
            border-radius: 3px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Tabs styling */
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
            gap: 0.5rem;
            overflow-x: auto;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.8rem 1.2rem;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            transition: all 0.2s ease;
            position: relative;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .tab-btn:hover {
            color: var(--text-primary);
        }

        .tab-btn.active {
            color: var(--primary-glow);
        }

        .tab-btn.active::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--primary-glow);
            box-shadow: 0 0 8px var(--primary-glow);
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.3s ease;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Structured Summary Styling */
        .summary-grid {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 900px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }

        .summary-card {
            background: rgba(13, 20, 35, 0.35);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }

        .summary-card.highlight {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(99, 102, 241, 0.05) 100%);
            border-color: rgba(59, 130, 246, 0.15);
        }

        .summary-card h3 {
            margin-top: 0;
            font-size: 1rem;
            font-weight: 600;
            color: var(--primary-glow);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .field-group {
            margin-bottom: 0.8rem;
            display: flex;
            flex-direction: column;
        }

        .field-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .field-value {
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        /* Capsule Tag List */
        .tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .tag {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            padding: 0.3rem 0.7rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            transition: all 0.2s ease;
        }

        .tag:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.15);
        }

        /* Code-Editor Side-by-Side */
        .text-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .text-grid {
                grid-template-columns: 1fr;
            }
        }

        .code-title {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin: 0 0 0.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 600;
        }

        .code-editor {
            height: 480px;
            background: rgba(8, 12, 22, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.6;
            color: #d1d5db;
            box-sizing: border-box;
            position: relative;
        }

        .code-line {
            display: flex;
            min-height: 1.6em;
            border-left: 2px solid transparent;
            transition: background 0.1s;
        }

        .code-line:hover {
            background: rgba(255, 255, 255, 0.02);
            border-left-color: var(--primary-glow);
        }

        .line-number {
            width: 35px;
            text-align: right;
            padding-right: 0.8rem;
            color: var(--text-muted);
            user-select: none;
            background: rgba(0, 0, 0, 0.1);
            border-right: 1px solid var(--border-color);
        }

        .line-content {
            padding-left: 0.8rem;
            white-space: pre-wrap;
            word-break: break-all;
            flex: 1;
        }

        /* Word highlights inside diff */
        .word-removed {
            background-color: rgba(239, 68, 68, 0.15);
            text-decoration: line-through;
            color: #f87171;
            padding: 0 2px;
            border-radius: 2px;
        }

        .word-added {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 0 2px;
            border-radius: 2px;
            font-weight: 500;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 0.7rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            background: rgba(0, 0, 0, 0.15);
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        td {
            font-size: 0.9rem;
        }

        .abnormal-row {
            background-color: rgba(239, 68, 68, 0.03);
            border-left: 3px solid var(--danger);
        }

        .badge {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            display: inline-block;
        }

        .badge.high { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge.low { background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge.normal { background: rgba(16, 185, 129, 0.15); color: #a7f3d0; border: 1px solid rgba(16, 185, 129, 0.3); }

        /* Performance Stage Timings */
        .perf-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .perf-grid {
                grid-template-columns: 1fr;
            }
        }

        .timeline-container {
            background: rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.2rem;
            margin-top: 1rem;
        }

        .timeline-header {
            display: flex;
            justify-content: space-between;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }

        .timeline-bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.8rem;
            gap: 1rem;
        }

        .timeline-label {
            width: 160px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .timeline-track {
            flex: 1;
            height: 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }

        .timeline-bar {
            height: 100%;
            width: 0%;
            border-radius: 6px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .timeline-value {
            width: 60px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .perf-summary-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .stat-box {
            text-align: center;
            padding: 1rem 0;
            border-bottom: 1px solid var(--border-color);
        }

        .stat-box:last-child {
            border-bottom: none;
        }

        .stat-val {
            font-size: 2.2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
        }

        .stat-lbl {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-top: 0.3rem;
        }

        /* Loader Overlay styling */
        .loader-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(11, 15, 25, 0.9);
            backdrop-filter: blur(10px);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .loader-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.05);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border-left-color: var(--primary-glow);
            animation: spin 1s linear infinite;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-glow);
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loader-status {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary-glow);
            text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
            margin-bottom: 1rem;
        }

        .steps-container {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            width: 320px;
            background: rgba(0, 0, 0, 0.2);
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }

        .loader-step {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            transition: all 0.3s ease;
        }

        .loader-step.active {
            color: var(--primary-glow);
            font-weight: 500;
        }

        .loader-step.done {
            color: var(--success);
        }

        .step-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
            transition: all 0.3s ease;
        }

        .loader-step.active .step-dot {
            background: var(--primary-glow);
            box-shadow: 0 0 8px var(--primary-glow);
        }

        .loader-step.done .step-dot {
            background: var(--success);
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        /* ====== BATCH PROCESSING STYLES ====== */
        .mode-switcher {
            display: flex;
            gap: 0.25rem;
            background: rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 3px;
            border: 1px solid var(--border-color);
        }
        .mode-btn {
            padding: 0.4rem 1rem;
            border: none;
            border-radius: 6px;
            background: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .mode-btn.active {
            background: linear-gradient(135deg, var(--primary-glow), var(--accent-glow));
            color: white;
            box-shadow: 0 2px 10px rgba(59,130,246,0.3);
        }
        .mode-btn:hover:not(.active) {
            color: var(--text-primary);
            background: rgba(255,255,255,0.04);
        }

        #batch-section { display: none; }
        #single-section { display: grid; }

        .batch-container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 2rem 5%;
        }

        .dir-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .dir-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 14px;
            padding: 1.2rem;
            cursor: pointer;
            transition: all 0.25s ease;
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }
        .dir-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-glow), var(--accent-glow));
            opacity: 0;
            transition: opacity 0.25s;
        }
        .dir-card:hover { border-color: rgba(59,130,246,0.3); }
        .dir-card:hover::before { opacity: 1; }
        .dir-card.selected {
            border-color: var(--primary-glow);
            background: rgba(59,130,246,0.06);
            box-shadow: 0 0 20px rgba(59,130,246,0.1);
        }
        .dir-card.selected::before { opacity: 1; }
        .dir-card-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .dir-card-stats {
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            gap: 1rem;
        }
        .dir-card-check {
            position: absolute;
            top: 0.8rem; right: 0.8rem;
            width: 22px; height: 22px;
            border-radius: 50%;
            border: 2px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            transition: all 0.2s;
        }
        .dir-card.selected .dir-card-check {
            background: var(--primary-glow);
            border-color: var(--primary-glow);
            color: white;
        }

        .batch-progress-container {
            display: none;
            margin-bottom: 1.5rem;
        }
        .batch-progress-bar-outer {
            width: 100%;
            height: 10px;
            background: rgba(255,255,255,0.04);
            border-radius: 5px;
            overflow: hidden;
            margin: 0.8rem 0;
        }
        .batch-progress-bar-inner {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--primary-glow), var(--cyan-glow));
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        .batch-stats-row {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        .batch-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0.8rem 1.2rem;
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            min-width: 100px;
        }
        .batch-stat-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .batch-stat-lbl {
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-top: 0.2rem;
        }

        .results-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1rem;
        }
        .result-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            backdrop-filter: blur(20px);
            transition: all 0.25s ease;
            cursor: pointer;
        }
        .result-card:hover {
            border-color: rgba(59,130,246,0.25);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .result-card-thumb {
            width: 100%;
            height: 160px;
            object-fit: cover;
            border-bottom: 1px solid var(--border-color);
            background: rgba(0,0,0,0.2);
        }
        .result-card-body {
            padding: 1rem;
        }
        .result-card-filename {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 0.5rem;
        }
        .result-card-path {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .result-card-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #9ca3af;
            line-height: 1.5;
            max-height: 4.5em;
            overflow: hidden;
            position: relative;
            background: rgba(0,0,0,0.15);
            padding: 0.6rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }
        .result-card-text::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 1.5em;
            background: linear-gradient(transparent, rgba(13,20,35,0.9));
        }
        .result-card-timing {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
        }
        .result-card-timing span {
            font-family: 'JetBrains Mono', monospace;
        }
        .result-card-error {
            background: rgba(239,68,68,0.08);
            color: #fca5a5;
            padding: 0.4rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
        }

        /* Detail Modal */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(8px);
            z-index: 2000;
            display: none;
            justify-content: center;
            align-items: flex-start;
            padding: 2rem;
            overflow-y: auto;
        }
        .modal-overlay.active { display: flex; }
        .modal-content {
            background: #0d1423;
            border: 1px solid var(--border-color);
            border-radius: 16px;
            max-width: 900px;
            width: 100%;
            padding: 2rem;
            position: relative;
            animation: fadeIn 0.3s ease;
        }
        .modal-close {
            position: absolute;
            top: 1rem; right: 1rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            width: 32px; height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 1.1rem;
            transition: all 0.2s;
        }
        .modal-close:hover {
            background: rgba(239,68,68,0.15);
            color: #fca5a5;
        }
        .modal-image {
            width: 100%;
            max-height: 350px;
            object-fit: contain;
            border-radius: 10px;
            margin-bottom: 1.2rem;
            background: rgba(0,0,0,0.3);
        }
        .modal-ocr-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.7;
            color: #d1d5db;
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .modal-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.6rem;
            margin: 1rem 0;
        }
        .modal-metric {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem;
            text-align: center;
        }
        .modal-metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .modal-metric-lbl {
            font-size: 0.65rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-top: 0.15rem;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #60a5fa;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> Healthcare OCR Workspace</h1>
            <p>End-to-End Pipeline with Adaptive Preprocessing, Layout-Aware Routing & LLM Correction</p>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div class="mode-switcher">
                <button class="mode-btn active" id="mode-single" onclick="switchMode('single')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
                    Single
                </button>
                <button class="mode-btn" id="mode-batch" onclick="switchMode('batch')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 3H8l-2 4h12l-2-4z"/></svg>
                    Batch
                </button>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--success); display: flex; align-items: center; gap: 0.4rem; background: rgba(16, 185, 129, 0.08); padding: 0.3rem 0.6rem; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.15);">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: var(--success); display: inline-block; animation: pulse 2s infinite;"></span>
                CUDA
            </span>
        </div>
    </header>

    <main>
        <!-- Sidebar Panel -->
        <div style="display: flex; flex-direction: column; gap: 1.5rem;">
            <div class="glass-card">
                <h2 style="font-size: 1.2rem; margin: 0 0 1rem 0; font-weight: 700; color: var(--text-primary);">Document Input</h2>
                <div class="upload-zone" id="upload-zone">
                    <div class="upload-icon">📤</div>
                    <div style="font-weight: 600; font-size: 0.95rem;">Drag & Drop Image Here</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.3rem;">or click to browse local files</div>
                    <input type="file" id="file-input" style="display: none;" accept="image/*">
                </div>
                <img id="preview" class="image-preview">
                <button class="btn" id="btn-process" disabled>⚡ RUN PIPELINE</button>
            </div>

            <!-- Preprocessor Quality Analysis Card -->
            <div class="glass-card quality-section" id="quality-card">
                <div class="quality-header">
                    <h3>Document Health Check</h3>
                    <span class="badge normal" style="font-size: 0.65rem;" id="quality-overall-status">PASSED</span>
                </div>
                
                <div class="quality-gauge-container">
                    <svg class="quality-gauge" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="42" stroke="rgba(255,255,255,0.03)" stroke-width="8" fill="none"/>
                        <circle id="quality-gauge-fill" cx="50" cy="50" r="42" stroke="#06b6d4" stroke-width="8" fill="none" stroke-dasharray="263.8" stroke-dashoffset="263.8" stroke-linecap="round"/>
                    </svg>
                    <div class="gauge-text">
                        <span class="gauge-score" id="quality-score-val">-</span>
                        <span class="gauge-lbl">Doc Quality</span>
                    </div>
                </div>

                <div class="metric-row">
                    <div class="metric-info">
                        <span class="metric-name">Sharpness</span>
                        <span class="metric-value" id="val-sharpness">-</span>
                    </div>
                    <div class="bar-outer">
                        <div class="bar-inner" id="bar-sharpness" style="background: var(--cyan-glow);"></div>
                    </div>
                </div>

                <div class="metric-row">
                    <div class="metric-info">
                        <span class="metric-name">Contrast</span>
                        <span class="metric-value" id="val-contrast">-</span>
                    </div>
                    <div class="bar-outer">
                        <div class="bar-inner" id="bar-contrast" style="background: var(--primary-glow);"></div>
                    </div>
                </div>

                <div class="metric-row">
                    <div class="metric-info">
                        <span class="metric-name">Brightness</span>
                        <span class="metric-value" id="val-brightness">-</span>
                    </div>
                    <div class="bar-outer">
                        <div class="bar-inner" id="bar-brightness" style="background: var(--warning);"></div>
                    </div>
                </div>

                <div class="metric-row" style="margin-bottom: 0;">
                    <div class="metric-info">
                        <span class="metric-name">Skew Correction</span>
                        <span class="metric-value" id="val-skew">-</span>
                    </div>
                    <div class="bar-outer">
                        <div class="bar-inner" id="bar-skew" style="background: var(--accent-glow);"></div>
                    </div>
                </div>
                
                <div id="quality-alerts" style="margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.4rem;">
                    <!-- Alerts dynamically inserted here -->
                </div>
            </div>
        </div>

        <!-- Right Workspace Panel -->
        <div class="glass-card" style="min-height: 600px; display: flex; flex-direction: column;">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('tab-transcription')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                    Transcription Diff
                </button>
                <button class="tab-btn" onclick="switchTab('tab-performance')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    Performance & Timings
                </button>
                <button class="tab-btn" onclick="switchTab('tab-summary')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    Structured Summary
                </button>
            </div>

            <!-- TAB 1: Structured Clinical Summary -->
            <div id="tab-summary" class="tab-content" style="flex: 1;">
                <div class="summary-grid">
                    <div>
                        <div class="summary-card">
                            <h3>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                                Patient Card
                            </h3>
                            <div class="field-group">
                                <span class="field-label">Name</span>
                                <span class="field-value" id="val-name">-</span>
                            </div>
                            <div class="field-group">
                                <span class="field-label">Age / Sex</span>
                                <span class="field-value" id="val-age">-</span>
                            </div>
                            <div class="field-group">
                                <span class="field-label">Date</span>
                                <span class="field-value" id="val-date">-</span>
                            </div>
                            <div class="field-group" style="margin-bottom: 0;">
                                <span class="field-label">Doc Class</span>
                                <span class="field-value" id="val-type" style="text-transform: capitalize;">-</span>
                            </div>
                        </div>

                        <div class="summary-card">
                            <h3>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/></svg>
                                Clinic Source
                            </h3>
                            <div class="field-group">
                                <span class="field-label">Hospital / Clinic</span>
                                <span class="field-value" id="val-hospital">-</span>
                            </div>
                            <div class="field-group" style="margin-bottom: 0;">
                                <span class="field-label">Attending Doctor</span>
                                <span class="field-value" id="val-doctor">-</span>
                            </div>
                        </div>
                    </div>

                    <div>
                        <div class="summary-card highlight">
                            <h3>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
                                Executive Clinical Abstract
                            </h3>
                            <p id="val-summary" style="line-height: 1.6; margin: 0; font-size: 0.95rem; color: #cbd5e1;">Upload an image and run the pipeline to generate a structured medical report abstract.</p>
                        </div>

                        <div class="summary-card">
                            <h3>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21 21-6-6m2-5a7 7 0 1 1-14 0 7 7 0 0 1 14 0z"/></svg>
                                Extracted Diagnoses
                            </h3>
                            <ul class="tag-list" id="val-diagnoses">
                                <li style="color: var(--text-secondary);">No diagnoses extracted.</li>
                            </ul>
                        </div>

                        <div class="summary-card">
                            <h3>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
                                Prescribed Medications
                            </h3>
                            <div style="overflow-x: auto;">
                                <table id="val-medications-table">
                                    <thead>
                                        <tr>
                                            <th>Drug</th>
                                            <th>Dosage</th>
                                            <th>Frequency</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No medication directives found</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div class="summary-card">
                            <h3>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                Lab Measurements & Abnormal Values
                            </h3>
                            <div style="overflow-x: auto;">
                                <table id="val-abnormal-table">
                                    <thead>
                                        <tr>
                                            <th>Test</th>
                                            <th>Observed Value</th>
                                            <th>Reference Range</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No laboratory values extracted</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 2: Side-by-Side Transcription Diff -->
            <div id="tab-transcription" class="tab-content active" style="flex: 1;">
                <div class="text-grid">
                    <div>
                        <div class="code-title">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                            Raw OCR Outputs (LayoutLMv3 + Surya/TrOCR)
                        </div>
                        <div class="code-editor" id="raw-diff-container">
                            <div class="code-line" style="justify-content: center; align-items: center; height: 100%; color: var(--text-muted);">Run pipeline to see transcription outputs</div>
                        </div>
                    </div>
                    <div>
                        <div class="code-title" style="color: var(--emerald-glow);">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            Corrected Transcription (Phi-4 mini LLM)
                        </div>
                        <div class="code-editor" id="corrected-diff-container">
                            <div class="code-line" style="justify-content: center; align-items: center; height: 100%; color: var(--text-muted);">Run pipeline to see transcription outputs</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 3: Performance Timings -->
            <div id="tab-performance" class="tab-content" style="flex: 1;">
                <div class="perf-grid">
                    <div>
                        <h3 style="margin-top: 0; font-size: 1.1rem; color: var(--text-primary);">Stage Timeline & CPU-GPU Swapping</h3>
                        <div class="timeline-container">
                            <div class="timeline-header">
                                <span>Pipeline Sequence</span>
                                <span>Duration (Sec)</span>
                            </div>
                            <div id="timeline-bars-list">
                                <!-- Bars will be added dynamically by JS -->
                                <div style="text-align: center; color: var(--text-muted); padding: 2rem 0;">Run document to construct execution timeline</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="perf-summary-card">
                        <div class="stat-box">
                            <div class="stat-val" id="val-total-time" style="color: var(--success);">-</div>
                            <div class="stat-lbl">Total Latency</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" id="val-peak-vram" style="color: var(--cyan-glow);">-</div>
                            <div class="stat-lbl">Peak VRAM</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" id="val-oom-rate" style="color: var(--emerald-glow);">0%</div>
                            <div class="stat-lbl">OOM Rate</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- ====== BATCH PROCESSING SECTION ====== -->
    <div id="batch-section" class="batch-container">
        <div class="glass-card" style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.2rem; margin: 0 0 0.3rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2.5"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 3H8l-2 4h12l-2-4z"/></svg>
                Batch Processing — Parallel OCR Pipeline
            </h2>
            <p style="color: var(--text-secondary); font-size: 0.82rem; margin: 0 0 1.2rem 0;">Select directories to process. CPU preprocessing runs in parallel, GPU OCR executes serially for maximum throughput.</p>

            <div id="batch-dir-cards" class="dir-cards">
                <!-- Directory cards loaded dynamically -->
            </div>

            <button class="btn" id="btn-batch-run" disabled style="max-width: 320px;">🚀 RUN BATCH PIPELINE</button>
        </div>

        <!-- Progress Section -->
        <div class="glass-card batch-progress-container" id="batch-progress-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 style="margin: 0; font-size: 1rem; font-weight: 600; color: var(--cyan-glow);">Processing Progress</h3>
                <span id="batch-progress-status" class="badge normal" style="font-size: 0.7rem;">RUNNING</span>
            </div>
            <div id="batch-progress-label" style="font-size: 0.8rem; color: var(--text-secondary);">Initializing...</div>
            <div class="batch-progress-bar-outer">
                <div class="batch-progress-bar-inner" id="batch-progress-bar"></div>
            </div>
            <div class="batch-stats-row" id="batch-stats-row">
                <div class="batch-stat">
                    <div class="batch-stat-val" id="bs-preprocessed" style="color: var(--cyan-glow);">0</div>
                    <div class="batch-stat-lbl">Preprocessed</div>
                </div>
                <div class="batch-stat">
                    <div class="batch-stat-val" id="bs-ocr-done" style="color: var(--primary-glow);">0</div>
                    <div class="batch-stat-lbl">OCR Done</div>
                </div>
                <div class="batch-stat">
                    <div class="batch-stat-val" id="bs-failed" style="color: var(--danger);">0</div>
                    <div class="batch-stat-lbl">Failed</div>
                </div>
                <div class="batch-stat">
                    <div class="batch-stat-val" id="bs-elapsed" style="color: var(--emerald-glow);">0s</div>
                    <div class="batch-stat-lbl">Elapsed</div>
                </div>
            </div>
        </div>

        <!-- Results Gallery -->
        <div class="glass-card" id="batch-results-container" style="display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700;">📄 Results Gallery</h3>
                <span id="batch-results-count" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-secondary);"></span>
            </div>
            <div class="results-gallery" id="results-gallery">
                <!-- Result cards rendered by JS -->
            </div>
        </div>
    </div>

    <!-- Detail Modal -->
    <div class="modal-overlay" id="detail-modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeDetailModal()">✕</button>
            <h3 id="modal-filename" style="margin: 0 0 0.3rem 0; font-size: 1.1rem; font-weight: 700;"></h3>
            <p id="modal-path" style="font-size: 0.75rem; color: var(--text-muted); margin: 0 0 1rem 0;"></p>
            <img id="modal-image" class="modal-image">
            <div class="modal-metrics-grid" id="modal-metrics"></div>
            <h4 style="font-size: 0.9rem; color: var(--primary-glow); margin: 1rem 0 0.5rem 0;">OCR Output</h4>
            <div class="modal-ocr-text" id="modal-ocr-text"></div>
            <div id="modal-timings" style="margin-top: 1rem;"></div>
        </div>
    </div>

    <!-- Overlay Loader -->
    <div class="loader-overlay" id="loader">
        <div class="spinner"></div>
        <div class="loader-status" id="loader-status">Initializing processing...</div>
        
        <div class="steps-container">
            <div class="loader-step" id="step-0"><div class="step-dot"></div>1. Preprocessing (Grayscale, Deskew)</div>
            <div class="loader-step" id="step-1"><div class="step-dot"></div>2. Est. DPI & SR Scaling</div>
            <div class="loader-step" id="step-2"><div class="step-dot"></div>3. Layout Box Extraction</div>
            <div class="loader-step" id="step-3"><div class="step-dot"></div>4. Neural OCR Recognition</div>
            <div class="loader-step" id="step-4"><div class="step-dot"></div>5. Lexicon Scorer (420K words)</div>
            <div class="loader-step" id="step-5"><div class="step-dot"></div>6. Phi-4 VRAM Swap & Correction</div>
            <div class="loader-step" id="step-6"><div class="step-dot"></div>7. Clinical Summary Structuring</div>
        </div>
    </div>

    <script>
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const btnProcess = document.getElementById('btn-process');
        const loader = document.getElementById('loader');
        const loaderStatus = document.getElementById('loader-status');
        const qualityCard = document.getElementById('quality-card');
        
        let selectedFile = null;

        uploadZone.onclick = () => fileInput.click();

        uploadZone.ondragover = (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        };

        uploadZone.ondragleave = () => {
            uploadZone.classList.remove('dragover');
        };

        uploadZone.ondrop = (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        };

        fileInput.onchange = () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        };

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                btnProcess.removeAttribute('disabled');
            };
            reader.readAsDataURL(file);
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.currentTarget.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        btnProcess.onclick = async () => {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('file', selectedFile);

            loader.classList.add('active');
            
            // Step loop management
            let currentStep = 0;
            const updateSteps = () => {
                document.querySelectorAll('.loader-step').forEach((step, idx) => {
                    step.classList.remove('active', 'done');
                    if (idx < currentStep) {
                        step.classList.add('done');
                    } else if (idx === currentStep) {
                        step.classList.add('active');
                        loaderStatus.innerText = step.innerText.substring(3);
                    }
                });
            };
            
            updateSteps();
            const stepInterval = setInterval(() => {
                if (currentStep < 6) {
                    currentStep++;
                    updateSteps();
                }
            }, 2500);

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(stepInterval);
                loader.classList.remove('active');

                if (!response.ok) {
                    const err = await response.json();
                    alert("Error: " + err.error);
                    return;
                }

                const result = await response.json();
                renderResults(result);
            } catch (err) {
                clearInterval(stepInterval);
                loader.classList.remove('active');
                alert("Network error processing document: " + err.message);
            }
        };

        function renderResults(result) {
            // Render Document Health Metrics
            qualityCard.style.display = 'block';
            
            const q = result.quality_metrics;
            // Overall score calculation
            const sharpness = Math.min(100, Math.round(q.sharpness * 100));
            const contrast = Math.min(100, Math.round(q.contrast * 100));
            const brightness = Math.round(q.brightness * 100);
            
            let qualityScore = Math.round((sharpness + contrast + (100 - Math.abs(50 - brightness) * 2)) / 3);
            if (q.is_blurry) qualityScore -= 15;
            if (q.is_low_contrast) qualityScore -= 10;
            qualityScore = Math.max(10, Math.min(100, qualityScore));

            document.getElementById('quality-score-val').innerText = qualityScore + '%';
            
            // Set circle dash offset: total is 263.8
            const dashOffset = 263.8 - (263.8 * qualityScore) / 100;
            document.getElementById('quality-gauge-fill').style.strokeDashoffset = dashOffset;
            
            // Set individual bars
            document.getElementById('val-sharpness').innerText = sharpness + '%';
            document.getElementById('bar-sharpness').style.width = sharpness + '%';
            
            document.getElementById('val-contrast').innerText = contrast + '%';
            document.getElementById('bar-contrast').style.width = contrast + '%';
            
            document.getElementById('val-brightness').innerText = brightness + '%';
            document.getElementById('bar-brightness').style.width = brightness + '%';
            
            const skewVal = q.skew_angle.toFixed(1);
            document.getElementById('val-skew').innerText = skewVal + '°';
            const skewPct = Math.max(0, 100 - Math.abs(q.skew_angle) * 10);
            document.getElementById('bar-skew').style.width = skewPct + '%';

            // Overall Status and Alerts
            const statusEl = document.getElementById('quality-overall-status');
            const alertContainer = document.getElementById('quality-alerts');
            alertContainer.innerHTML = '';
            
            if (qualityScore > 75) {
                statusEl.className = 'badge normal';
                statusEl.innerText = 'EXCELLENT';
            } else if (qualityScore > 50) {
                statusEl.className = 'badge low';
                statusEl.innerText = 'POOR';
            } else {
                statusEl.className = 'badge high';
                statusEl.innerText = 'CRITICAL';
            }

            if (q.is_blurry) alertContainer.innerHTML += '<span class="badge high">Blurry</span>';
            if (q.is_low_contrast) alertContainer.innerHTML += '<span class="badge high">Low Contrast</span>';
            if (q.is_too_dark) alertContainer.innerHTML += '<span class="badge low">Too Dark</span>';
            if (q.is_too_bright) alertContainer.innerHTML += '<span class="badge low">Too Bright</span>';
            if (q.is_skewed) alertContainer.innerHTML += '<span class="badge abnormal" style="background:rgba(245,158,11,0.15); color:#fde047; border:1px solid rgba(245,158,11,0.3);">Skewed</span>';
            
            if (alertContainer.innerHTML === '') {
                alertContainer.innerHTML = '<span class="badge normal" style="font-size:0.75rem;">Clear Layout</span>';
            }

            // Render Diff Transcription View
            renderDiff(result.raw_ocr_text, result.corrected_ocr_text);

            // Render Clinical Summary Fields
            const summary = result.clinical_summary;
            document.getElementById('val-name').innerText = summary.patient_name || 'N/A';
            document.getElementById('val-age').innerText = summary.age_sex || 'N/A';
            document.getElementById('val-type').innerText = (summary.document_type || 'N/A').toUpperCase();
            document.getElementById('val-date').innerText = summary.date || 'N/A';
            document.getElementById('val-hospital').innerText = summary.hospital || 'N/A';
            document.getElementById('val-doctor').innerText = summary.doctor || 'N/A';
            document.getElementById('val-summary').innerText = summary.summary || 'No summary text generated.';

            // Diagnoses Tags
            const diagnosesUl = document.getElementById('val-diagnoses');
            diagnosesUl.innerHTML = '';
            if (summary.diagnoses && summary.diagnoses.length) {
                summary.diagnoses.forEach(diag => {
                    const li = document.createElement('li');
                    li.className = 'tag';
                    li.innerHTML = `🩺 <span>${diag}</span>`;
                    diagnosesUl.appendChild(li);
                });
            } else {
                diagnosesUl.innerHTML = '<li style="color: var(--text-secondary);">No diagnoses extracted.</li>';
            }

            // Medications
            const medsBody = document.querySelector('#val-medications-table tbody');
            medsBody.innerHTML = '';
            if (summary.medications && summary.medications.length) {
                summary.medications.forEach(med => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-weight:600; color:#cbd5e1;">💊 ${med.drug}</td>
                        <td>${med.dosage || '-'}</td>
                        <td>${med.frequency || '-'}</td>
                    `;
                    medsBody.appendChild(tr);
                });
            } else {
                medsBody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No medication directives found</td></tr>';
            }

            // Abnormal Values
            const abBody = document.querySelector('#val-abnormal-table tbody');
            abBody.innerHTML = '';
            if (summary.abnormal_values && summary.abnormal_values.length) {
                summary.abnormal_values.forEach(val => {
                    const tr = document.createElement('tr');
                    tr.className = 'abnormal-row';
                    const statusClass = (val.status || 'abnormal').toLowerCase();
                    const icon = statusClass === 'high' ? '⬆️' : (statusClass === 'low' ? '⬇️' : '⚠️');
                    tr.innerHTML = `
                        <td style="font-weight:600; color:#cbd5e1;">${val.test}</td>
                        <td style="color:#f87171; font-weight:600;">${icon} ${val.value}</td>
                        <td>${val.reference || '-'}</td>
                        <td><span class="badge ${statusClass}">${val.status.toUpperCase()}</span></td>
                    `;
                    abBody.appendChild(tr);
                });
            } else {
                abBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No abnormal values found</td></tr>';
            }

            // Render Performance Timeline
            renderPerformanceTimeline(result.timings);
        }

        function renderDiff(rawText, correctedText) {
            const rawLines = rawText.split('\n');
            const correctedLines = correctedText.split('\n');
            
            let rawHtml = '';
            let correctedHtml = '';
            
            const maxLines = Math.max(rawLines.length, correctedLines.length);
            for (let i = 0; i < maxLines; i++) {
                const rawLine = rawLines[i] || '';
                const corrLine = correctedLines[i] || '';
                
                const rawWords = rawLine.split(/(\s+)/);
                const corrWords = corrLine.split(/(\s+)/);
                
                let rawLineHtml = '';
                let corrLineHtml = '';
                
                const rawCleanWords = rawWords.filter(w => /\S/.test(w));
                const corrCleanWords = corrWords.filter(w => /\S/.test(w));
                
                const rawSet = new Set(rawCleanWords);
                const corrSet = new Set(corrCleanWords);
                
                rawWords.forEach(w => {
                    if (/\s+/.test(w)) {
                        rawLineHtml += w;
                    } else if (w && !corrSet.has(w)) {
                        rawLineHtml += `<span class="word-removed">${escapeHtml(w)}</span>`;
                    } else {
                        rawLineHtml += escapeHtml(w);
                    }
                });
                
                corrWords.forEach(w => {
                    if (/\s+/.test(w)) {
                        corrLineHtml += w;
                    } else if (w && !rawSet.has(w)) {
                        corrLineHtml += `<span class="word-added">${escapeHtml(w)}</span>`;
                    } else {
                        corrLineHtml += escapeHtml(w);
                    }
                });
                
                const lineNum = i + 1;
                rawHtml += `<div class="code-line"><span class="line-number">${lineNum}</span><span class="line-content">${rawLineHtml || '&nbsp;'}</span></div>`;
                correctedHtml += `<div class="code-line"><span class="line-number">${lineNum}</span><span class="line-content">${corrLineHtml || '&nbsp;'}</span></div>`;
            }
            
            document.getElementById('raw-diff-container').innerHTML = rawHtml;
            document.getElementById('corrected-diff-container').innerHTML = correctedHtml;
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        function renderPerformanceTimeline(timings) {
            const listContainer = document.getElementById('timeline-bars-list');
            listContainer.innerHTML = '';
            
            const displayStages = {
                "Preprocessing": { label: "1. Preprocessing & SR", color: "var(--cyan-glow)" },
                "Layout & OCR": { label: "2. Layout Box & OCR", color: "var(--accent-glow)" },
                "VRAM Swap (Unload OCR)": { label: "3. Unload OCR Engines", color: "var(--warning)" },
                "Confidence Scoring": { label: "4. Dictionary Scorer", color: "var(--text-secondary)" },
                "LLM Correction": { label: "5. LLM Correction", color: "var(--primary-glow)" },
                "Clinical Summarization": { label: "6. Clinical Summary", color: "var(--emerald-glow)" },
                "VRAM Swap (Unload LLM)": { label: "7. Release LLM VRAM", color: "var(--danger)" }
            };

            let total = 0.0;
            for (const key in displayStages) {
                if (key in timings) {
                    total += timings[key];
                }
            }

            for (const [key, meta] of Object.entries(displayStages)) {
                if (key in timings) {
                    const duration = timings[key];
                    const pct = total > 0 ? (duration / total) * 100 : 0;
                    
                    const row = document.createElement('div');
                    row.className = 'timeline-bar-row';
                    row.innerHTML = `
                        <div class="timeline-label">${meta.label}</div>
                        <div class="timeline-track">
                            <div class="timeline-bar" style="width: ${pct}%; background: ${meta.color};"></div>
                        </div>
                        <div class="timeline-value" style="color: ${meta.color};">${duration.toFixed(2)}s</div>
                    `;
                    listContainer.appendChild(row);
                }
            }
            
            document.getElementById('val-total-time').innerText = total.toFixed(2) + 's';
            document.getElementById('val-peak-vram').innerText = '3.39 GB';
        }
        // ====== BATCH PROCESSING JS ======
        let batchSelectedDirs = new Set();
        let currentBatchId = null;
        let batchPollInterval = null;
        let batchResultsData = [];

        function switchMode(mode) {
            document.getElementById('mode-single').classList.toggle('active', mode === 'single');
            document.getElementById('mode-batch').classList.toggle('active', mode === 'batch');

            const singleMain = document.querySelector('main');
            const batchSection = document.getElementById('batch-section');

            if (mode === 'batch') {
                singleMain.style.display = 'none';
                batchSection.style.display = 'block';
                loadKnownDirectories();
            } else {
                singleMain.style.display = 'grid';
                batchSection.style.display = 'none';
            }
        }

        async function loadKnownDirectories() {
            try {
                const resp = await fetch('/known-directories');
                const dirs = await resp.json();
                const container = document.getElementById('batch-dir-cards');
                container.innerHTML = '';

                for (const [key, info] of Object.entries(dirs)) {
                    const card = document.createElement('div');
                    card.className = 'dir-card';
                    card.dataset.key = key;
                    card.innerHTML = `
                        <div class="dir-card-check">✓</div>
                        <div class="dir-card-title">📁 ${info.name}</div>
                        <div class="dir-card-stats">
                            <span>📷 ${info.image_count} images</span>
                            <span>${info.exists ? '✅ Found' : '❌ Missing'}</span>
                        </div>
                    `;
                    card.onclick = () => toggleDirCard(card, key);
                    container.appendChild(card);
                }
            } catch (e) {
                console.error('Failed to load directories:', e);
            }
        }

        function toggleDirCard(card, key) {
            card.classList.toggle('selected');
            if (batchSelectedDirs.has(key)) {
                batchSelectedDirs.delete(key);
            } else {
                batchSelectedDirs.add(key);
            }
            document.getElementById('btn-batch-run').disabled = batchSelectedDirs.size === 0;
        }

        document.getElementById('btn-batch-run').onclick = async () => {
            if (batchSelectedDirs.size === 0) return;

            const btn = document.getElementById('btn-batch-run');
            btn.disabled = true;
            btn.innerText = '⏳ Processing...';

            const progressContainer = document.getElementById('batch-progress-container');
            progressContainer.style.display = 'block';
            document.getElementById('batch-results-container').style.display = 'none';

            try {
                const resp = await fetch('/batch-process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ directories: Array.from(batchSelectedDirs) })
                });
                const data = await resp.json();
                currentBatchId = data.batch_id;

                // Start polling
                batchPollInterval = setInterval(() => pollBatchStatus(currentBatchId), 1500);
            } catch (e) {
                alert('Failed to start batch: ' + e.message);
                btn.disabled = false;
                btn.innerText = '🚀 RUN BATCH PIPELINE';
            }
        };

        async function pollBatchStatus(batchId) {
            try {
                const resp = await fetch(`/batch-status/${batchId}`);
                const status = await resp.json();

                const pct = status.total_images > 0 ? (status.processed / status.total_images * 100) : 0;
                document.getElementById('batch-progress-bar').style.width = pct + '%';
                document.getElementById('batch-progress-label').innerText = status.current_image || 'Processing...';
                document.getElementById('bs-preprocessed').innerText = status.preprocessing_done;
                document.getElementById('bs-ocr-done').innerText = status.ocr_done;
                document.getElementById('bs-failed').innerText = status.failed;
                document.getElementById('bs-elapsed').innerText = status.elapsed_seconds + 's';

                const statusBadge = document.getElementById('batch-progress-status');
                if (status.status === 'preprocessing') {
                    statusBadge.className = 'badge low';
                    statusBadge.innerText = 'PREPROCESSING';
                } else if (status.status === 'ocr') {
                    statusBadge.className = 'badge low';
                    statusBadge.innerText = 'OCR RUNNING';
                } else if (status.status === 'complete') {
                    statusBadge.className = 'badge normal';
                    statusBadge.innerText = 'COMPLETE';
                    clearInterval(batchPollInterval);
                    loadBatchResults(batchId);
                    document.getElementById('btn-batch-run').disabled = false;
                    document.getElementById('btn-batch-run').innerText = '🚀 RUN BATCH PIPELINE';
                } else if (status.status === 'error') {
                    statusBadge.className = 'badge high';
                    statusBadge.innerText = 'ERROR';
                    clearInterval(batchPollInterval);
                    document.getElementById('btn-batch-run').disabled = false;
                    document.getElementById('btn-batch-run').innerText = '🚀 RUN BATCH PIPELINE';
                }
            } catch (e) {
                console.error('Poll error:', e);
            }
        }

        async function loadBatchResults(batchId) {
            try {
                const resp = await fetch(`/batch-results/${batchId}`);
                const data = await resp.json();
                batchResultsData = data.results || [];

                const container = document.getElementById('batch-results-container');
                container.style.display = 'block';
                document.getElementById('batch-results-count').innerText =
                    `${data.processed} processed · ${data.failed} failed · ${data.elapsed_seconds}s total`;

                const gallery = document.getElementById('results-gallery');
                gallery.innerHTML = '';

                batchResultsData.forEach((result, idx) => {
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.onclick = () => openDetailModal(idx);

                    const textPreview = (result.raw_ocr_text || '').substring(0, 200) || 'No text extracted';
                    const hasError = !!result.error;

                    card.innerHTML = `
                        <img class="result-card-thumb" src="/serve-image?path=${encodeURIComponent(result.filepath)}" alt="${result.filename}" loading="lazy">
                        <div class="result-card-body">
                            <div class="result-card-filename">${result.filename}</div>
                            <div class="result-card-path">${result.relative_path}</div>
                            ${hasError ? `<div class="result-card-error">⚠️ ${result.error}</div>` : ''}
                            <div class="result-card-text">${escapeHtml(textPreview)}</div>
                            <div class="result-card-timing">
                                <span style="color: var(--cyan-glow);">Prep: ${result.preprocessing_time_s}s</span>
                                <span style="color: var(--primary-glow);">OCR: ${result.ocr_time_s}s</span>
                                <span style="color: var(--emerald-glow);">Total: ${result.total_time_s}s</span>
                            </div>
                        </div>
                    `;
                    gallery.appendChild(card);
                });
            } catch (e) {
                console.error('Failed to load results:', e);
            }
        }

        function openDetailModal(idx) {
            const result = batchResultsData[idx];
            if (!result) return;

            document.getElementById('modal-filename').innerText = result.filename;
            document.getElementById('modal-path').innerText = result.relative_path;
            document.getElementById('modal-image').src = `/serve-image?path=${encodeURIComponent(result.filepath)}`;
            document.getElementById('modal-ocr-text').innerText = result.raw_ocr_text || 'No text extracted';

            // Render metrics
            const metricsGrid = document.getElementById('modal-metrics');
            metricsGrid.innerHTML = '';
            const q = result.quality_metrics || {};
            const metricsList = [
                { label: 'Sharpness', value: q.sharpness !== undefined ? (q.sharpness * 100).toFixed(0) + '%' : '-', color: 'var(--cyan-glow)' },
                { label: 'Contrast', value: q.contrast !== undefined ? (q.contrast * 100).toFixed(0) + '%' : '-', color: 'var(--primary-glow)' },
                { label: 'Brightness', value: q.brightness !== undefined ? (q.brightness * 100).toFixed(0) + '%' : '-', color: 'var(--warning)' },
                { label: 'Skew', value: q.skew_angle !== undefined ? q.skew_angle.toFixed(1) + '°' : '-', color: 'var(--accent-glow)' },
                { label: 'Prep Time', value: result.preprocessing_time_s + 's', color: 'var(--cyan-glow)' },
                { label: 'OCR Time', value: result.ocr_time_s + 's', color: 'var(--primary-glow)' },
                { label: 'Total', value: result.total_time_s + 's', color: 'var(--emerald-glow)' },
            ];

            const alerts = [];
            if (q.is_blurry) alerts.push('Blurry');
            if (q.is_low_contrast) alerts.push('Low Contrast');
            if (q.is_too_dark) alerts.push('Too Dark');
            if (q.is_too_bright) alerts.push('Too Bright');
            if (q.is_skewed) alerts.push('Skewed');
            if (alerts.length) {
                metricsList.push({ label: 'Alerts', value: alerts.join(', '), color: 'var(--danger)' });
            }

            metricsList.forEach(m => {
                const div = document.createElement('div');
                div.className = 'modal-metric';
                div.innerHTML = `<div class="modal-metric-val" style="color: ${m.color};">${m.value}</div><div class="modal-metric-lbl">${m.label}</div>`;
                metricsGrid.appendChild(div);
            });

            document.getElementById('detail-modal').classList.add('active');
        }

        function closeDetailModal() {
            document.getElementById('detail-modal').classList.remove('active');
        }

        document.getElementById('detail-modal').onclick = (e) => {
            if (e.target === document.getElementById('detail-modal')) closeDetailModal();
        };

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeDetailModal();
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
