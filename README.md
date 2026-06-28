# Medical Report OCR + Summarization Pipeline

A high-speed medical document OCR + summarization pipeline that processes patient-uploaded medical reports (lab reports, handwritten prescriptions, discharge summaries, ultrasound reports, case booklets) and produces structured, doctor-readable summaries.

## Setup

1. Create a Python 3.12 virtual environment:
   ```bash
   py -3.12 -m venv venv
   ```
2. Activate and install dependencies:
   ```bash
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run verification:
   ```bash
   python scripts/verify_env.py
   ```
