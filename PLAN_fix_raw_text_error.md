# Plan - Fix raw_text UnboundLocalError

## SECTION A — GOAL DEFINITION
1. **What is being built or changed?**
   - We are fixing an `UnboundLocalError` in the batch processing runner (`src/batch_runner.py`) where `raw_text` is accessed in the progress print statement but is occasionally unbound/not associated with a value.
2. **What does "done" look like?**
   - The batch runner completes successfully without raising `UnboundLocalError`, and progress/results are correctly logged/stored for all preprocessed and OCR'd images.
3. **What is explicitly out of scope?**
   - Modifying the OCR engine selection or preprocessing logic.
   - Adjusting layout detection or other pipeline features.

## SECTION B — TECH STACK
- Python 3.14 (running the server environment)
- [batch_runner.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/batch_runner.py) (The file containing the bug)

## SECTION C — SESSION MODULARIZATION
### Session 1: Code Fix & Verification
- **Objective**: Fix the `UnboundLocalError` by initializing `raw_text` at the very beginning of the image processing loop iteration.
- **Scope**: [src/batch_runner.py](file:///C:/Users/oliad/Desktop/HealthCareOCR/src/batch_runner.py)
- **Output**: A modified batch runner that handles `raw_text` safely on all code paths (including errors/skips).
- **Connects to**: Server restart and verification testing.
- **Failure Surface**: Syntax or indentation error during editing.

## SECTION D — PROGRESS CHECKLIST
- [ ] Session 1: Code Fix & Verification
  - [x] Initialize `raw_text = ""` at the start of the `for prep_result in preprocessed_queue:` loop in `_run_batch_sync`.
  - [x] Verify that the server starts up correctly.
  - [ ] Run a test batch process to confirm successful completion without `UnboundLocalError`.
