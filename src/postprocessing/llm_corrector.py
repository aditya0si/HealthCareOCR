import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.gpu_manager import GPUManager


class LLMCorrector:
    """
    Medical OCR post-processing corrector using Phi-4 Mini Instruct.
    
    Loads microsoft/Phi-4-mini-instruct in BF16, then applies INT4 weight-only
    quantization at runtime using torchao 0.7.0. This avoids the torchao nightly
    dependency required by the pre-quantized pytorch/Phi-4-mini-instruct-AWQ-INT4.
    
    Expected VRAM: ~2.5 GB (INT4 quantized)
    """
    
    BASE_MODEL_ID = "microsoft/Phi-4-mini-instruct"

    def __init__(self, model_id: str = None, device: str = "cuda"):
        self.model_id = model_id or self.BASE_MODEL_ID
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Lazy-loads the tokenizer and model using bitsandbytes 4-bit quantization,
        or moves it back to CUDA if it was swapped to CPU.
        """
        if self.model is not None:
            try:
                device_type = next(self.model.parameters()).device.type
                if device_type == "cpu":
                    print("Moving LLM back to CUDA...")
                    self.model.to(self.device)
            except Exception as e:
                print(f"Error checking/moving LLM device: {e}")
            return
            
        print(f"Loading LLM {self.model_id} in 4-bit...")
        GPUManager.print_vram_status("Before loading LLM")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Configure bitsandbytes 4-bit
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model directly in 4-bit to GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            attn_implementation="sdpa"
        )
        self.model.eval()
        
        GPUManager.print_vram_status("After loading LLM")
        print("LLM loaded successfully.")

    def correct_text(self, text_with_uncertain_markers: str) -> str:
        """
        Runs correction on lines containing [UNCERTAIN] using the LLM in a single batch.
        """
        if not text_with_uncertain_markers:
            return ""
            
        # Ensure model is loaded
        self.load_model()
        
        lines = text_with_uncertain_markers.split("\n")
        uncertain_line_indices = []
        uncertain_lines = []
        
        for idx, line in enumerate(lines):
            if "[UNCERTAIN]" in line:
                uncertain_line_indices.append(idx)
                uncertain_lines.append(line)
                
        if not uncertain_lines:
            return text_with_uncertain_markers.replace("[UNCERTAIN]", "")
            
        # Build prompt for each line
        prompts = []
        for line in uncertain_lines:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical spelling correction assistant. "
                        "Correct only the [UNCERTAIN] word(s) in the input line using medical context. "
                        "Remove the [UNCERTAIN] prefix from the corrected word. "
                        "Return only the corrected line, preserving the rest of the text exactly. "
                        "Do not explain, do not add filler, and do not repeat the prompt or prefix."
                    )
                },
                {
                    "role": "user",
                    "content": f"Input line: {line}"
                }
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            
        # Tokenize with padding
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|end|>")]
            )
            
        # Decode and replace
        for i, idx in enumerate(uncertain_line_indices):
            gen_ids = outputs[i][input_len:]
            corrected_line = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            corrected_line = corrected_line.split("\n")[0].strip()
            
            # Clean common prefixes
            for prefix in ["line to correct:", "input line:", "corrected line:", "corrected:", "line:"]:
                if corrected_line.lower().startswith(prefix):
                    corrected_line = corrected_line[len(prefix):].strip()
            corrected_line = corrected_line.replace("`", "").strip()
            
            if not corrected_line:
                corrected_line = uncertain_lines[i].replace("[UNCERTAIN]", "")
                
            lines[idx] = corrected_line
            
        return "\n".join(lines)

    def unload(self):
        """
        Moves the model to CPU to release GPU VRAM.
        """
        if self.model is not None:
            print("Moving LLM to CPU...")
            try:
                self.model.to("cpu")
            except Exception as e:
                print(f"Error moving LLM to CPU: {e}")
            GPUManager.clean_memory()
            GPUManager.print_vram_status("After unloading LLM (swapped to CPU)")
