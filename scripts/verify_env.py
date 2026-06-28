import sys
import os
import time

def verify_environment():
    print("=" * 60)
    print("MEDICAL REPORT OCR PIPELINE - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # 1. Check Python version
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Executable: {sys.executable}")
    print("-" * 60)
    
    # 2. Check imports of critical dependencies
    dependencies = {
        "torch": "PyTorch (Deep Learning Core)",
        "torchvision": "Torchvision (Image Transforms)",
        "transformers": "Hugging Face Transformers",
        "bitsandbytes": "bitsandbytes (NF4 Quantization)",
        "accelerate": "Hugging Face Accelerate",
        "surya": "Surya OCR (Printed text OCR)",
        "onnxruntime": "ONNX Runtime (LayoutLMv3 Inference)",
        "cv2": "OpenCV (Image I/O & Bounding Boxes)",
        "skimage": "Scikit-Image (Sauvola Binarization)",
        "deskew": "Deskew library (Orientation)",
        "sentencepiece": "SentencePiece (Tokenizer)",
        "yaml": "PyYAML (Configuration parsing)",
        "pymedtermino": "PyMedTermino (Medical Lexicon)",
        "einops": "Einops (Tensor operations)"
    }
    
    print("Checking Dependency Imports:")
    all_ok = True
    for module_name, desc in dependencies.items():
        try:
            start_time = time.time()
            __import__(module_name)
            elapsed = time.time() - start_time
            print(f"  [OK] {module_name:<15} - {desc} (Loaded in {elapsed:.3f}s)")
        except ImportError as e:
            print(f"  [FAIL] {module_name:<15} - {desc} (Error: {e})")
            all_ok = False
            
    print("-" * 60)
    
    # 3. Check CUDA & PyTorch GPU integration
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA Device Count: {device_count}")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  Device {i}: {name}")
                print(f"    Compute Capability: {cap[0]}.{cap[1]}")
                print(f"    Total VRAM: {total_mem:.2f} GB")
            
            # Simple GPU test operation
            print("Running GPU Compute Sanity Test...")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"  GPU MatMul (1000x1000) took {elapsed * 1000:.3f} ms. Compute is working!")
            
            # Print CUDA memory summary
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            cached = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"  Allocated VRAM: {allocated:.2f} MB, Reserved/Cached: {cached:.2f} MB")
        else:
            print("  [WARNING] PyTorch cannot access GPU. Verify CUDA driver installation.")
            all_ok = False
    except Exception as e:
        print(f"  [ERROR] Failed to run PyTorch GPU verification: {e}")
        all_ok = False
        
    print("=" * 60)
    if all_ok:
        print("VERIFICATION STATUS: ALL PASS! Environment is ready.")
    else:
        print("VERIFICATION STATUS: FAILED/WARNINGS. Resolve missing dependencies above.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
