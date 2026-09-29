import torch
import torch.nn as nn
from pathlib import Path

from config.train_config import(
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    NUM_CLASSES,
)

from src.ml.dataset import create_dataloaders
from src.ml.model import PVFaultModel
from src.ml.preprocessing import (
    get_eval_transform,
    get_train_transform,
)
from src.ml.trainer import Trainer

DATASET_PATH = "data/InfraredSolarModules"

def main(model_save_path: str | Path = "weights/drone_v2"):
    model_save_path = Path(model_save_path)
    if DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # torchvision expects image size as (height, width)
    image_size = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    
    # Create preprocessing pipelines
    train_transform = get_train_transform(
        image_size=image_size
    )

    eval_transform = get_eval_transform(
        image_size=image_size
    )
    
    # Create train / validation / test dataloaders
    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = create_dataloaders(
        dataset_path=DATASET_PATH,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=BATCH_SIZE,
        dataset_format="metadata"
    )

    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    if len(class_names) != NUM_CLASSES : 
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(class_names)}."
        )
        
    #create the mobilenetv3 model
    resume = (model_save_path / "last.pt").is_file()
    model = PVFaultModel(
        num_classes=len(class_names),
        pretrained=not resume,
        freeze_backbone=False,
    )
    
    # Classification loss
    criterion = nn.CrossEntropyLoss()
    #optimization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_directory=model_save_path

    )
    
    # Start training
    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=EPOCHS,
        class_names=class_names,
        resume=resume,
    )
    
    print("Training complete.")

    return history


if __name__ == "__main__":
    main(model_save_path="weights/drone_optuna_1")
