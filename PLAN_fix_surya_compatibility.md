# Plan - Fix Surya-Transformers Compatibility and OCR Engine Loading

## SECTION A — GOAL DEFINITION
1. **What is being built or changed?**
   - Downgrade the `transformers` library in Python 3.14 to version `< 5.0` to resolve `AttributeError: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'`.
   - Modify `load_ocr_engines` in [src/pipeline.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py) to check if `self.router` is `None` (in addition to `self.layout`), resolving the `AttributeError: 'NoneType' object has no attribute 'process_document'` error that occurs when initialization fails partially.
2. **What does "done" look like?**
   - The server warm-up completes successfully.
   - Batch processes and single image processes complete without `AttributeError` or `NoneType` errors.
3. **What is explicitly out of scope?**
   - Upgrading/modifying layout engine models or other logic.

## SECTION B — TECH STACK
- Python 3.14
- `transformers<5.0` (specifically downgrading from `5.12.1` to a stable `4.x` version)
- [src/pipeline.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py)

## SECTION C — SESSION MODULARIZATION
### Session 1: Dependency Downgrade & Code Fix
- **Objective**: Downgrade `transformers` and apply robustness fixes to `load_ocr_engines`.
- **Scope**: [src/pipeline.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/pipeline.py) and Python 3.14 environment dependencies.
- **Output**: Python 3.14 with `transformers<5.0` and a robust `load_ocr_engines` implementation.
- **Connects to**: Server restart and verification testing.
- **Failure Surface**: Dependency installation failures or syntax/logic bugs in `src/pipeline.py`.

## SECTION D — PROGRESS CHECKLIST
- [ ] Session 1: Dependency Downgrade & Code Fix
  - [ ] Install `transformers<5.0` in Python 3.14 environment.
  - [ ] Modify `load_ocr_engines` in `src/pipeline.py` to check for `self.router is None`.
  - [ ] Verify that the server starts up and completes warm-up run successfully.
  - [ ] Run a test batch process to confirm successful completion of OCR text extraction.
