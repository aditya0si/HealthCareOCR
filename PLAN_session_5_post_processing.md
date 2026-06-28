# PLAN: Session 5 — OCR Post-Processing + Conditional LLM Correction

## SECTION A — GOAL DEFINITION

The goal of Session 5 is to build the post-processing stage of our pipeline. It scores the confidence of the extracted raw OCR text using a combination of character/word recognition confidence and a compiled medical terminology dictionary. It then triggers conditional spelling and semantic correction using the Phi-4 Mini 3.8B AWQ model, but only on regions flagged as low-confidence.

### Observable Outcomes ("Done" Criteria):
1. **Medical Dictionary Loader (`medical_dict.py`)** downloads and merges standard English wordlists (~370K terms) with a medical wordlist (~98K terms) and Indian medical abbreviations.
2. **Confidence Scorer (`confidence_scorer.py`)** flags low-confidence words (lookup score < 0.6) using exact matching and fuzzy Levenshtein distance matching.
3. **Medical Tokenizer (`tokenizer.py`)** loads or trains a SentencePiece tokenizer to prepare text for LLM correction.
4. **VRAM Swapping & Lazy Loader (`gpu_manager.py`)**: Unloads OCR engines (Surya + TrOCR), clears GPU memory, and loads Phi-4 Mini AWQ safely without triggering out-of-memory (OOM) errors.
5. **Conditional LLM Corrector (`llm_corrector.py`)**: Checks for flagged words. If none, skips LLM correction (0ms). If present, sends text with `[UNCERTAIN]` markers to Phi-4 Mini AWQ for deterministic correction.
6. **Execution Speed & Memory**: LLM load and forward pass on corrections complete in **< 3.0s** on the RTX 5060 GPU.

### Out of Scope:
- End-to-end multi-image processing (restricted to single-image files).
- Layout structural modifications.

---

## SECTION B — TECH STACK

- **PyTorch (CUDA 13.2)**: GPU compute framework for running deep learning models.
- **transformers / autoawq / vllm**: For loading and running Phi-4 Mini 3.8B AWQ.
- **pymedtermino**: Medical terminology helper.
- **urllib**: To download public English and medical wordlists.
- **rapidfuzz / Levenshtein**: For high-performance fuzzy dictionary matching.

---

## SECTION C — SESSION MODULARIZATION

### Sub-session 5.1: Medical & English Lexicon Compiler (`medical_dict.py`)
- **Objective:** Build a robust, caching lookup dictionary containing standard English words, medical terms, abbreviations, and units.
- **Details:** Downloads and parses standard lists, caching them in `resources/` to avoid repeated network overhead.

### Sub-session 5.2: Dictionary Confidence Scorer (`confidence_scorer.py`)
- **Objective:** Evaluate OCR text word-by-word, flagging misspelled or low-confidence words for correction.
- **Interface:** `ConfidenceScorer.score_text(text: str) -> tuple[str, list[dict]]` (returns text with `[UNCERTAIN]` tags and metadata).

### Sub-session 5.3: VRAM Memory Orchestrator & Lazy Loader (`gpu_manager.py`)
- **Objective:** Manage transition from OCR models (Surya + TrOCR) to LLM (Phi-4 Mini AWQ) inside the 8GB limit.
- **Implementation:** Explicitly garbage-collects model weights, empties PyTorch caching allocator, and loads/unloads models sequentially.

### Sub-session 5.4: Conditional LLM Corrector (`llm_corrector.py`)
- **Objective:** Load Phi-4 Mini AWQ, format prompt with marked uncertain words, run correction, and clean up.
- **Interface:** `LLMCorrector.correct_text(text_with_markers: str) -> str`

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 5: OCR Post-Processing + Conditional LLM Correction**
  - [ ] Implement `src/utils/medical_dict.py` (caching dictionary builder)
  - [ ] Implement `src/postprocessing/confidence_scorer.py` (word confidence scoring + fuzzy matching)
  - [ ] Create `src/utils/gpu_manager.py` (Sequential model manager and cache clearer)
  - [ ] Implement `src/postprocessing/llm_corrector.py` (Phi-4 Mini AWQ conditional corrector)
  - [ ] Write tests in `tests/test_postprocessing.py` verifying dictionary loading, confidence scoring, VRAM swapping, and correction quality
  - [ ] Verify that model swapping unloads OCR engines and loads Phi-4 Mini under 8GB VRAM limit (< 3.2 GB peak VRAM)
  - [ ] Verify that conditional correction completes in < 3.0 seconds
