"""
Custom PyTorch Dataset for loading DICOM files directly
"""

import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms

class CheXpertDICOMDataset(Dataset):
    """
    PyTorch Dataset for CheXpert that loads DICOM files on-the-fly
    """

    def __init__(self, csv_file, transform=None, target_size=(224, 224)):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Target size for images (height, width)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size

        # CheXpert label columns
        self.label_columns = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Lesion",
            "Lung Opacity",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get DICOM file path
        dicom_path = self.data_frame.iloc[idx]['dicom_path']

        # Load DICOM file
        try:
            ds = pydicom.dcmread(dicom_path)
            image_array = ds.pixel_array.astype(float)

            # Normalize to 0-255
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8) * 255.0
            image_array = image_array.astype(np.uint8)

            # Convert to PIL Image (grayscale)
            image = Image.fromarray(image_array)

            # Convert grayscale to RGB (3 channels) for pretrained models
            image = image.convert('RGB')

        except Exception as e:
            print(f"Error loading {dicom_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', self.target_size, (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        labels = self.data_frame.iloc[idx][self.label_columns].values.astype('float32')
        labels = torch.from_numpy(labels)

        # Get metadata
        subject_id = self.data_frame.iloc[idx]['subject_id']
        study_id = self.data_frame.iloc[idx]['study_id']

        sample = {
            'image': image,
            'labels': labels,
            'subject_id': subject_id,
            'study_id': study_id,
            'dicom_path': dicom_path
        }

        return sample


def get_data_loaders(train_csv, val_csv, test_csv, batch_size=16, num_workers=4):
    """
    Create data loaders for train, validation, and test sets

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # No augmentation for validation/test
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CheXpertDICOMDataset(train_csv, transform=train_transform)
    val_dataset = CheXpertDICOMDataset(val_csv, transform=val_transform)
    test_dataset = CheXpertDICOMDataset(test_csv, transform=val_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[OK] Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import os

    DATASET_DIR = r"E:\download\dataset"
    train_csv = os.path.join(DATASET_DIR, "train.csv")

    print("Testing CheXpertDICOMDataset...")
    dataset = CheXpertDICOMDataset(train_csv)

    print(f"Dataset size: {len(dataset)}")

    # Test loading one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Labels: {sample['labels']}")
    print(f"  Subject ID: {sample['subject_id']}")
    print(f"  Study ID: {sample['study_id']}")
    print(f"  DICOM path: {sample['dicom_path']}")

    print("\n[OK] Dataset test passed!")
