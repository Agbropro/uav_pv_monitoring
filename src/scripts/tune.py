from pathlib import Path

import optuna
import torch
import torch.nn as nn

from config.train_config import DEVICE, NUM_CLASSES
from src.ml.dataset import create_dataloaders
from src.ml.model import PVFaultModel
from src.ml.preprocessing import get_eval_transform, get_train_transform
from src.ml.trainer import Trainer


def main(
    dataset_path: str | Path,
    model_save_path: str | Path,
    n_trials: int,
    epochs: int,
):
    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        #Objective function focusses on minimizing best val loss
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        image_size = trial.suggest_categorical(
            "image_size", ["48x80", "72x120", "96x160"]
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [False, True])
        height, width = (int(dimension) for dimension in image_size.split("x"))
        train_transform = get_train_transform(image_size=(height, width))
        eval_transform = get_eval_transform(image_size=(height, width))

        train_loader, val_loader, _, class_names = create_dataloaders(
            dataset_path=str(dataset_path),
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=batch_size,
            dataset_format="metadata",
        )
        if len(class_names) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} classes, but found {len(class_names)}")

        model = PVFaultModel(
            num_classes=len(class_names),
            pretrained=True,
            freeze_backbone=freeze_backbone,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        trial_path = model_save_path / f"trial_{trial.number}"
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            save_directory=trial_path,
        )

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = trainer.train_epoch(train_loader)
            val_loss, val_accuracy = trainer.validate(val_loader)
            print(
                f"Trial {trial.number} | Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer._save_checkpoint(
                    path=trial_path / "best.pt",
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    class_names=class_names,
                )

            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_loss

    study = optuna.create_study(
        study_name="mobilenet_v3_small",
        storage=f"sqlite:///{(model_save_path / 'study.db').resolve()}",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Best checkpoint: {model_save_path / f'trial_{study.best_trial.number}' / 'best.pt'}")
    return study


if __name__ == "__main__":
    main(
        dataset_path="data/InfraredSolarModules",
        model_save_path="weights/optuna_mobilenet_v3_small",
        n_trials=20,
        epochs=20,
    )
