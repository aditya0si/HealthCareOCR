import os
import argparse
import torch
import logging
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, get_scheduler, VisionEncoderDecoderConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.training.data_loader import get_dataloader

# Monkeypatch to avoid AttributeError: 'VisionEncoderDecoderConfig' object has no attribute 'vocab_size' in peft
def get_vocab_size(self):
    return self.decoder.vocab_size if hasattr(self, 'decoder') else getattr(self, '_vocab_size', None)

def set_vocab_size(self, val):
    self._vocab_size = val

VisionEncoderDecoderConfig.vocab_size = property(get_vocab_size, set_vocab_size)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Medical OCR Model Fine-Tuning Pipeline (LoRA)")
    parser.add_argument("--dataset_dir", type=str, default="medical-prescription-dataset", help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-base-handwritten", help="Base model checkpoint name")
    parser.add_argument("--save_dir", type=str, default="models/adapters/trocr_lora", help="Directory to save LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max target sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    return parser.parse_args()

def configure_lora_trocr(model):
    """
    Configures LoRA for TrOCR (VisionEncoderDecoderModel).
    Targets self-attention layers in the encoder (ViT) and self/cross attention in the decoder (RoBERTa).
    """
    # Based on our analysis, encoder self-attention has query, value, and decoder attention has q_proj, v_proj
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value", "q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=[]
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model

def train(args):
    logger.info(f"Starting training run on device: {args.device}")
    
    # Load processor
    logger.info(f"Loading processor: {args.model_name}")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    
    # Load base model
    logger.info(f"Loading base model: {args.model_name}")
    base_model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    
    # Set decoder start token id, pad token id, and eos token id
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.vocab_size = base_model.config.decoder.vocab_size
    
    # Apply LoRA wrapping
    logger.info("Applying PEFT/LoRA wrapper...")
    model = configure_lora_trocr(base_model)
    model.to(args.device)
    
    # Enable gradient checkpointing to save VRAM
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    
    # Log trainable parameter details
    model.print_trainable_parameters()
    
    # Load datasets/dataloaders
    logger.info("Initializing DataLoaders...")
    train_loader = get_dataloader(
        dataset_dir=args.dataset_dir,
        split="train",
        processor=processor,
        batch_size=args.batch_size,
        max_target_length=args.max_length,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        dataset_dir=args.dataset_dir,
        split="val",
        processor=processor,
        batch_size=args.batch_size,
        max_target_length=args.max_length,
        shuffle=False
    )
    
    # Define optimizer
    # Select 8-bit AdamW optimizer if bitsandbytes is available to save memory
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=args.lr)
        logger.info("Initialized bitsandbytes 8-bit AdamW optimizer.")
    except Exception as e:
        logger.warning(f"Could not load bitsandbytes 8-bit optimizer: {e}. Falling back to standard PyTorch AdamW.")
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        
    num_training_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # PyTorch mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training Loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        pbar = tqdm(train_loader, desc="Training")
        
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(args.device)
            labels = batch["labels"].to(args.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                # Normalize loss for gradient accumulation
                loss = loss / args.grad_accum
                
            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum
            
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(args.device)
                labels = batch["labels"].to(args.device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save best adapter
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            logger.info(f"Saving new best LoRA adapter checkpoint to {args.save_dir}")
            model.save_pretrained(args.save_dir)
            
    logger.info("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    train(args)
