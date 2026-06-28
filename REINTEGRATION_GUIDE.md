# Reintegration Guide: Restoring LLM Correction & Clinical Summarization

This document provides step-by-step instructions on how to seamlessly reintegrate the `microsoft/Phi-4-mini-instruct` LLM correction and clinical summarization features back into the pipeline and user interface. 

---

## Part 1: Reintegration in the Pipeline (`src/pipeline.py`)

Locate the bypassed stages in the `process_image` method of `src/pipeline.py` and replace the commented-out block with the active execution logic.

### File: `src/pipeline.py`

#### Replace this bypassed block:
```python
        # Stage 4: Confidence Scoring (Bypassed for speed and to avoid LLM CUDA usage)
        # with timer("Confidence Scoring"):
        #     marked_text, flagged_words = self.scorer.process_text(raw_text)
            
        # Stage 5: Conditional LLM Correction (Bypassed for speed and to avoid LLM CUDA usage)
        corrected_text = raw_text
            
        summary_json = {}
        # if not skip_summarization:
        #     ...
```

#### With the original active LLM stages:
```python
        # Stage 4: Confidence Scoring
        with timer("Confidence Scoring"):
            marked_text, flagged_words = self.scorer.process_text(raw_text)
            
        # Stage 5: Conditional LLM Correction
        corrected_text = raw_text
        if len(flagged_words) > 0:
            with timer("LLM Correction"):
                # Load Phi-4 model (either from scratch or move CPU -> CUDA)
                self.corrector.load_model()
                corrected_text = self.corrector.correct_text(marked_text)
        else:
            print("No low-confidence words flagged. Skipping LLM correction.")
            
        summary_json = {}
        if not skip_summarization:
            # Stage 6: Clinical Summarization
            with timer("Clinical Summarization"):
                # Always ensure the LLM is loaded/moved to CUDA via corrector
                self.corrector.load_model()
                
                # Share loaded Phi-4 model with summarizer
                self.summarizer.model = self.corrector.model
                self.summarizer.tokenizer = self.corrector.tokenizer
                    
                summary_json = self.summarizer.summarize_text(corrected_text)
                
            # Stage 7: VRAM Swap (Unload LLM)
            with timer("VRAM Swap (Unload LLM)"):
                # Move the shared LLM to CPU
                self.corrector.unload()
                self.summarizer.model = None
                self.summarizer.tokenizer = None
                GPUManager.clean_memory()
        else:
            # If we skipped summarization but loaded the LLM for correction, we must still unload it!
            if len(flagged_words) > 0:
                with timer("VRAM Swap (Unload LLM)"):
                    self.corrector.unload()
                    self.summarizer.model = None
                    self.summarizer.tokenizer = None
                    GPUManager.clean_memory()
```

---

## Part 2: Reintegration in the Web UI (`app.py`)

Re-enable the tabs, side-by-side diff layout, loading screen steps, and step timer in the UI.

### File: `app.py`

#### 1. Tabs Menu Layout
Replace the hidden tab button and single-column title in `app.py`:
```html
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('tab-transcription')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                    OCR Transcription
                </button>
                <button class="tab-btn" onclick="switchTab('tab-performance')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    Performance & Timings
                </button>
                <button class="tab-btn" style="display: none;" onclick="switchTab('tab-summary')">
                    ...
                </button>
            </div>
```
With the original structure showing the Structured Summary tab:
```html
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
```

#### 2. Side-by-Side Diff Panel
Replace the single column raw layout container:
```html
            <!-- TAB 2: OCR Transcription Outputs -->
            <div id="tab-transcription" class="tab-content active" style="flex: 1;">
                <div class="text-grid" style="grid-template-columns: 1fr;">
                    <div>
                        <div class="code-title">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                            Raw OCR Outputs (LayoutLMv3 + Surya/TrOCR)
                        </div>
                        <div class="code-editor" id="raw-diff-container">
                            <div class="code-line" style="justify-content: center; align-items: center; height: 100%; color: var(--text-muted);">Run pipeline to see transcription outputs</div>
                        </div>
                    </div>
                    <!-- Hidden container to keep JS from throwing errors -->
                    <div id="corrected-diff-container" style="display: none;"></div>
                </div>
            </div>
```
With the original side-by-side Layout and Correction panels:
```html
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
```

#### 3. Loader Overlay Steps
Remove the inline styles (`style="display: none;"`) from steps 6 and 7 in the steps container:
```html
            <div class="loader-step" id="step-5" style="display: none;"><div class="step-dot"></div>6. Phi-4 VRAM Swap & Correction</div>
            <div class="loader-step" id="step-6" style="display: none;"><div class="step-dot"></div>7. Clinical Summary Structuring</div>
```
Restoring them to standard visible divs:
```html
            <div class="loader-step" id="step-5"><div class="step-dot"></div>6. Phi-4 VRAM Swap & Correction</div>
            <div class="loader-step" id="step-6"><div class="step-dot"></div>7. Clinical Summary Structuring</div>
```

#### 4. Step Timer Max Limit
Change the timer condition in the JS block back to its original limit:
```javascript
            const stepInterval = setInterval(() => {
                if (currentStep < 6) { // originally was < 4
                    currentStep++;
                    updateSteps();
                }
            }, 2500);
```

---

## Part 3: Running the Application with CUDA Support

CUDA runtime errors often occur when running standard pip environments that download PyTorch without matching CUDA libraries. 
To ensure CUDA works out-of-the-box, use the global system python environment which contains the pre-compiled CUDA-capable PyTorch build (`2.12.0+cu130`):

```powershell
# Set PYTHONPATH to the current workspace root
$env:PYTHONPATH="."

# Run using system python instead of the venv python
python app.py
```
