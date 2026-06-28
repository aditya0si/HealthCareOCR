import asyncio
import os
import sys
import json
from playwright.async_api import async_playwright

# Selected test images representing different document styles:
# 1. Printed Lab Report
# 2. Handwritten Prescription
# 3. Mixed layout/Case booklet
TEST_IMAGES = {
    "printed_lab_report": "Patient_Kastoor/Lab_Report/20260612_110755.jpg",
    "handwritten_prescription": "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10/handwritten_prescription_01.jpeg",
    "mixed_case_booklet": "WhatsApp.Unknown.2026-04-27.at.12.10.10/WhatsApp Unknown 2026-04-27 at 12.10.10/discharge_summary_case_booklet_01.jpeg"
}

async def run_ocr_on_file(playwright, file_path):
    print(f"\nEvaluating: {file_path}")
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"Error: File not found at {abs_path}")
        return None
        
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    
    try:
        # Navigate to the local web app
        await page.goto("http://127.0.0.1:7860/")
        
        # Set file input directly
        file_input = page.locator("#file-input")
        await file_input.set_input_files(abs_path)
        
        # Verify preview image loads and process button becomes enabled
        await page.wait_for_selector("#btn-process:not([disabled])")
        
        # Wait for the API response while clicking the button
        async with page.expect_response("**/process", timeout=180000) as response_info:
            await page.click("#btn-process")
            
        response = await response_info.value
        if not response.ok:
            print(f"Error: HTTP Status {response.status} from API")
            return None
            
        result_json = await response.json()
        await browser.close()
        return result_json
        
    except Exception as e:
        print(f"Error executing browser automation for {file_path}: {e}")
        await browser.close()
        return None

async def main():
    print("Starting Playwright automation evaluation...")
    results = {}
    
    async with async_playwright() as playwright:
        for doc_type, img_path in TEST_IMAGES.items():
            res = await run_ocr_on_file(playwright, img_path)
            if res:
                results[doc_type] = {
                    "image": img_path,
                    "raw_ocr_text": res.get("raw_ocr_text", ""),
                    "corrected_ocr_text": res.get("corrected_ocr_text", ""),
                    "quality_metrics": res.get("quality_metrics", {}),
                    "timings": res.get("timings", {})
                }
                
    # Save the output to a JSON report for analysis
    output_report = "playwright_ocr_eval_report.json"
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAutomation complete. Saved report to: {output_report}")

if __name__ == "__main__":
    asyncio.run(main())
