import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)

class PrescriptionDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str = "train", processor = None, max_target_length: int = 128):
        """
        PyTorch Dataset for the Medical Prescription OCR Dataset.
        
        Args:
            dataset_dir: Path to the dataset root folder (e.g. medical-prescription-dataset).
            split: train, val, or test.
            processor: TrOCRProcessor instance.
            max_target_length: Maximum target sequence length.
        """
        self.split = split
        self.dataset_dir = os.path.join(dataset_dir, split)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        self.processor = processor
        self.max_target_length = max_target_length

        if not os.path.exists(self.images_dir) or not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Dataset folders for split '{split}' not found at: {self.dataset_dir}")

        # Get list of images
        self.image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])
        logger.info(f"Loaded {len(self.image_filenames)} examples for split '{split}' from {self.dataset_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load corresponding annotation
        base_name = os.path.splitext(img_name)[0]
        annotation_path = os.path.join(self.annotations_dir, f"{base_name}.json")
        
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        
        ground_truth = annotation["ground_truth"]
        
        # Load image using PIL
        image = Image.open(img_path).convert("RGB")
        
        # Preprocess using TrOCRProcessor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Tokenize labels
        labels = self.processor.tokenizer(
            ground_truth,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Replace padding token ids with -100 so that PyTorch's CrossEntropyLoss ignores them
        labels = torch.where(labels == self.processor.tokenizer.pad_token_id, -100, labels)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def get_dataloader(dataset_dir: str, split: str, processor, batch_size: int = 2, max_target_length: int = 128, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Creates and returns a DataLoader for the PrescriptionDataset.
    """
    dataset = PrescriptionDataset(
        dataset_dir=dataset_dir,
        split=split,
        processor=processor,
        max_target_length=max_target_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
