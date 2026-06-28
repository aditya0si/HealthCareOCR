# PLAN: Summarizer Optimization & Benchmarking

## SECTION A — GOAL DEFINITION

The goal of this task is to optimize the clinical summarizer pipeline to reduce generation latency, prevent hallucination loops, and verify the pipeline meets all timing (P50 <= 5.0s, P90 <= 7.0s) and memory constraints (peak VRAM <= 4.0GB).

### Questions Answered:
1. **What is being built or changed?**
   - The clinical summarizer prompt and system instruction in `src/summarization/summarizer.py` will be modified to request a flat, single-line JSON omitting empty keys, and the parser will be updated to programmatically restore all missing keys.
2. **What does "done" look like?**
   - Summarizer latency drops to <= 3 seconds.
   - No JSON parsing errors or runaway generation.
   - End-to-end benchmark succeeds with P50 latency <= 5.0 seconds and P90 latency <= 7.0 seconds.
3. **What is explicitly out of scope?**
   - Retraining or fine-tuning models.
   - Changing the image preprocessing pipeline.

---

## SECTION B — TECH STACK

- **Languages/Runtime**: Python 3.12, PyTorch 2.12.1+cu132, CUDA 13.2
- **Models**: Microsoft Phi-4-mini-instruct (4-bit bitsandbytes NF4)
- **Files Modified**:
  - [summarizer.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/src/summarization/summarizer.py)
- **Files Touched/Run**:
  - [benchmark.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/scripts/benchmark.py)

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Prompt & Parsing Optimization
- **Objective**: Optimize prompt to restrict output tokens and restore schema keys programmatically in the parser.
- **Scope**: `src/summarization/summarizer.py`
- **Output**: Optimized prompt requesting compressed, flat, single-line JSON without empty/null keys, and updated parser filling missing keys with default schema values (`None` or `[]`).
- **Connects to**: Session 2 (Verification).
- **Failure Surface**: Malformed JSON outputs or missing expected keys.

### Session 2: Benchmarking and Validation
- **Objective**: Execute the dataset-wide benchmark script and verify latency and memory performance.
- **Scope**: `scripts/benchmark.py`, `task.md`, `walkthrough.md`
- **Output**: Passing benchmarks with P50 <= 5.0s, P90 <= 7.0s, VRAM <= 4GB, and 0% OOM rate.
- **Failure Surface**: Pipeline execution timeout or VRAM exhaustion.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 1: Prompt & Parsing Optimization**
  - [ ] Modify summarizer prompt in `src/summarization/summarizer.py`
  - [ ] Implement robust `_parse_json_response` to inject default keys for omitted ones
  - [ ] Verify summarizer output on sample medical texts
- [ ] **Session 2: Benchmarking and Validation**
  - [ ] Run `scripts/benchmark.py` over target images
  - [ ] Confirm P50 latency <= 5.0 seconds
  - [ ] Confirm P90 latency <= 7.0 seconds
  - [ ] Verify Peak Memory <= 4.0 GB VRAM and OOM rate is 0%
  - [ ] Update `task.md` and `walkthrough.md` with final results
