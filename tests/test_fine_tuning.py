import os
import tempfile
import torch
import pytest
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from peft import PeftModel
from src.training.data_loader import PrescriptionDataset
from src.training.trainer import configure_lora_trocr
from src.ocr.trocr_engine import TrOCREngine
from src.ocr.surya_engine import SuryaEngine

# Monkeypatch to avoid AttributeError: 'VisionEncoderDecoderConfig' object has no attribute 'vocab_size' in peft
def get_vocab_size(self):
    return self.decoder.vocab_size if hasattr(self, 'decoder') else getattr(self, '_vocab_size', None)

def set_vocab_size(self, val):
    self._vocab_size = val

VisionEncoderDecoderConfig.vocab_size = property(get_vocab_size, set_vocab_size)

@pytest.mark.skipif(not os.path.exists("medical-prescription-dataset"), reason="medical-prescription-dataset directory not found")
def test_prescription_dataset_loading():
    """
    Test that the PrescriptionDataset correctly loads data and preprocesses images and text.
    """
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    
    dataset = PrescriptionDataset(
        dataset_dir="medical-prescription-dataset",
        split="train",
        processor=processor,
        max_target_length=64
    )
    
    assert len(dataset) > 0, "Dataset should have at least one item."
    
    # Load first item
    item = dataset[0]
    assert "pixel_values" in item
    assert "labels" in item
    assert isinstance(item["pixel_values"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["pixel_values"].shape == (3, 384, 384), "TrOCR pixel_values shape mismatch."
    assert item["labels"].shape == (64,), "Labels shape mismatch."

@pytest.mark.skipif(not os.path.exists("medical-prescription-dataset"), reason="medical-prescription-dataset directory not found")
def test_lora_wrapping_and_step_execution():
    """
    Test that TrOCR can be wrapped with LoRA, execute a forward/backward pass, 
    and save/load adapters cleanly.
    """
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    
    # Load dataset
    dataset = PrescriptionDataset(
        dataset_dir="medical-prescription-dataset",
        split="train",
        processor=processor,
        max_target_length=32
    )
    item = dataset[0]
    
    # Load base model on CPU/CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.vocab_size = base_model.config.decoder.vocab_size
    
    # Wrap model with LoRA
    lora_model = configure_lora_trocr(base_model)
    lora_model.to(device)
    
    # Trainable parameters check
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    logger_percentage = 100 * trainable_params / total_params
    print(f"\nTrainable params: {trainable_params} / {total_params} ({logger_percentage:.4f}%)")
    assert trainable_params > 0, "LoRA should make some parameters trainable."
    
    # Execute single forward/backward pass
    pixel_values = item["pixel_values"].unsqueeze(0).to(device)
    labels = item["labels"].unsqueeze(0).to(device)
    
    outputs = lora_model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss
    assert loss is not None
    assert not torch.isnan(loss)
    
    loss.backward()
    
    # Verify gradients exist in LoRA weights
    has_gradients = False
    for name, param in lora_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_gradients = True
            break
    assert has_gradients, "Active parameters should receive gradients after backward pass."

    # Test adapter saving & dynamic loading in TrOCREngine
    with tempfile.TemporaryDirectory() as tmp_dir:
        lora_model.save_pretrained(tmp_dir)
        
        # Verify saved files
        config_path = os.path.join(tmp_dir, "adapter_config.json")
        assert os.path.exists(config_path), "adapter_config.json not found."
        
        # Verify dynamic load in TrOCREngine
        engine = TrOCREngine(config={
            "model_name": model_name,
            "lora_dir": tmp_dir,
            "device": "cpu"  # CPU for test isolation
        })
        
        assert isinstance(engine.model, PeftModel), "Engine model should be wrapped in PeftModel."
        
        # Ensure we can run recognize on a dummy cropped image
        dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
        # Bounding box covering whole dummy image
        res = engine.recognize(dummy_img, [[0, 0, 200, 100]])
        assert isinstance(res, list)
        assert len(res) == 1
        assert isinstance(res[0], str)
