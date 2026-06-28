# PLAN: Session 7 — Fine-Tuning Infrastructure

## SECTION A — GOAL DEFINITION

The goal of Session 7 is to build the fine-tuning infrastructure for the local OCR models (specifically TrOCR for handwritten prescriptions and optionally Surya for layout/printed text) to improve transcription accuracy on medical handwriting. Due to strict local hardware constraints (RTX 5060 Laptop, 8GB VRAM), all fine-tuning must be highly resource-efficient, utilizing parameter-efficient fine-tuning (LoRA), 8-bit optimizers, and gradient checkpointing to prevent Out-Of-Memory (OOM) exceptions.

### Observable Outcomes ("Done" Criteria):
1. **LoRA Layer Configuration**: A script/module defining LoRA targets for TrOCR (`microsoft/trocr-base-handwritten` encoder/decoder query and value projections) and Surya.
2. **Medical Prescription Dataset Loader (`src/training/data_loader.py`)**: A custom dataset loader that parses `medical-prescription-dataset` JSON annotations and loads/preprocesses corresponding PIL images.
3. **Training Loop Orchestrator (`src/training/trainer.py`)**: A resource-optimized PyTorch training loop incorporating:
   - Gradient checkpointing (`model.gradient_checkpointing_enable()`)
   - 8-bit AdamW optimizer (`bitsandbytes.optim.Adam8bit` or similar) to save VRAM
   - Mixed-precision training (`torch.cuda.amp.autocast` / GradScaler or standard PyTorch FP16/BF16)
   - Dynamic batch size with gradient accumulation
4. **Adapter Serialization & Lifecycle**: Ability to save trained LoRA adapter weights (`adapter_model.safetensors`/`bin`) and load them dynamically into our runtime `TrOCREngine` and `SuryaEngine`.
5. **Accuracy Verification**: Test script demonstrating successful training step execution, saving of weights, and successful loading of fine-tuned adapters back into the inference engines.

### Out of Scope:
- Massive multi-epoch full model pre-training.
- Cloud-based training platforms or multi-GPU training.
- Training the Phi-4 LLM corrector (which is kept as-is in 4-bit).

---

## SECTION B — TECH STACK

- **Core DL Framework**: PyTorch (with AMP / gradient checkpointing)
- **Quantization/Optimization**: `bitsandbytes` (for 8-bit AdamW and memory-efficient parameters)
- **Fine-Tuning Utilities**: `peft` (Parameter-Efficient Fine-Tuning for LoRA configuration)
- **Hugging Face Hub / Tools**: `transformers` (`VisionEncoderDecoderModel` for TrOCR)
- **Target Models**:
  - `microsoft/trocr-base-handwritten` (Vision-Encoder Decoder, LoRA applied to encoder self-attention & decoder self-attention / cross-attention)
  - Surya2 recognition model (`SuryaModel`, LoRA applied to attention query/key/value projections)
- **Dataset**: `medical-prescription-dataset` (Local PNG image files + JSON labels mapping to `<s_ocr>` ground truths)

---

## SECTION C — SESSION MODULARIZATION

### Session 7.1: PEFT/LoRA Integration & Configuration
- **Objective**: Integrate `peft` and design LoRA configurations for both TrOCR and Surya.
- **Details**: Configure target modules (e.g. `q_proj`, `v_proj` in decoder self-attention/encoder self-attention) and verify parameters trainable ratio.
- **Output**: Python function `get_lora_model(model, task_type)` returning a trainable `PeftModel`.
- **Failure Surface**: Incorrect module names causing PEFT to find 0 modules, or model casting bugs.

### Session 7.2: Dataset Loader (`src/training/data_loader.py`)
- **Objective**: Implement a PyTorch `Dataset` and `DataLoader` to parse the local prescription dataset.
- **Details**: Loads images and processes them using TrOCR's processor (`TrOCRProcessor`) and Surya's foundation model processor, generating input pixel values and label token IDs.
- **Output**: Train/Val PyTorch dataloaders yield batched training features.
- **Failure Surface**: Large images causing high CPU memory load or preprocessing latency bottlenecking the GPU.

### Session 7.3: Optimized Trainer Loop (`src/training/trainer.py`)
- **Objective**: Implement the training runner with strict VRAM safeguards.
- **Details**: Integrates 8-bit AdamW optimizer, enables gradient checkpointing, and utilizes gradient accumulation. Saves checkpoints.
- **Output**: Running training command executing without OOM on 8GB GPU.
- **Failure Surface**: CUDA Out-of-Memory during backward pass if gradient checkpointing or gradient accumulation is misconfigured.

### Session 7.4: Dynamic Adapter Loading in Inference Engines
- **Objective**: Modify `TrOCREngine` and `SuryaEngine` to optionally load fine-tuned LoRA weights.
- **Details**: Loads base model, then calls `PeftModel.from_pretrained(base_model, adapter_dir)` if adapters exist.
- **Output**: Engines recognize and transcribe using fine-tuned weights when present.
- **Failure Surface**: Mismatched model types or shape mismatch during adapter loading.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 7: Fine-Tuning Infrastructure**
  - [ ] Install `peft` and verify package availability
  - [ ] Implement `src/training/data_loader.py` for parsing `medical-prescription-dataset`
  - [ ] Implement `src/training/trainer.py` supporting TrOCR/Surya LoRA configurations and 8-bit AdamW + gradient checkpointing
  - [ ] Add argument parsing to trainer script to run training on a subset of data for quick validation
  - [ ] Write integration test verifying LoRA model creation, training step execution, save, and reload
  - [ ] Adapt `TrOCREngine` and `SuryaEngine` to dynamically load PEFT LoRA adapters from a specified directory
  - [ ] Run evaluation script comparing base vs fine-tuned model OCR accuracy
