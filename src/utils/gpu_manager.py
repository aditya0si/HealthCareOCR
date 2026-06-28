import gc
import torch

class GPUManager:
    _instances = {}

    @classmethod
    def get_best_gpu_device(cls) -> str:
        """
        Selects the best available GPU, preferring discrete NVIDIA GPUs over integrated graphics.
        Returns a device string like "cuda:0" or "cuda:1".
        Falls back to "cpu" if no GPU available.
        """
        if not torch.cuda.is_available():
            print("[GPU] No CUDA devices available. Using CPU.")
            return "cpu"
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return "cpu"
        
        # Try to find discrete GPU (RTX, GTX, A100, etc.) - avoid Intel Arc, iGPU
        best_device = 0
        best_score = -1
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i).lower()
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            
            # Score: prefer discrete GPUs, higher VRAM
            score = 0
            
            # Strong preference for RTX/GTX (discrete NVIDIA)
            if "rtx" in gpu_name or "gtx" in gpu_name:
                score += 1000
            # Also prefer other professional NVIDIA cards
            elif "a100" in gpu_name or "a40" in gpu_name or "tesla" in gpu_name:
                score += 900
            # Penalize integrated graphics and Arc GPUs
            elif "intel" in gpu_name or "arc" in gpu_name or "integrated" in gpu_name:
                score -= 500
            
            # Add VRAM as secondary factor
            score += total_mem * 10
            
            print(f"[GPU] Device {i}: {gpu_name} ({total_mem:.1f}GB) - Score: {score}")
            
            if score > best_score:
                best_score = score
                best_device = i
        
        selected = f"cuda:{best_device}"
        print(f"[GPU] Selected device: {selected} ({torch.cuda.get_device_name(best_device)})")
        return selected

    @classmethod
    def get_vram_allocated(cls) -> float:
        """
        Returns currently allocated CUDA memory in MB.
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0

    @classmethod
    def print_vram_status(cls, label: str):
        """
        Prints the current VRAM allocation details.
        """
        if torch.cuda.is_available():
            allocated = cls.get_vram_allocated()
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"[VRAM Status - {label}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        else:
            print(f"[VRAM Status - {label}] CUDA not available")

    @classmethod
    def clean_memory(cls):
        """
        Performs aggressive garbage collection and releases cached PyTorch CUDA allocator blocks.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def unload_model(cls, model_obj):
        """
        Safely deletes references to a model, moves it to CPU, and clears memory.
        """
        if model_obj is not None:
            # Attempt to move weights to CPU to break CUDA references
            if hasattr(model_obj, "model") and model_obj.model is not None:
                try:
                    model_obj.model.to("cpu")
                    del model_obj.model
                except Exception:
                    pass
            if hasattr(model_obj, "detector") and model_obj.detector is not None:
                try:
                    if hasattr(model_obj.detector, "model"):
                        model_obj.detector.model.to("cpu")
                    del model_obj.detector
                except Exception:
                    pass
            del model_obj
            cls.clean_memory()
