from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    *,
    device: torch.device | str = "cpu",
    criterion: nn.Module | None = None,
) -> float:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.train()
    loss_sum = 0.0
    count = 0
    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError("criterion must return a finite scalar loss")
        loss.backward()
        optimizer.step()
        batch_size = targets.shape[0]
        loss_sum += loss.item() * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("Training batches are empty")
    return loss_sum / count


def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> dict[str, float | Tensor]:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    was_training = model.training
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    loss_sum = 0.0
    count = 0
    try:
        with torch.inference_mode():
            for images, targets in batches:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                if logits.ndim != 2 or logits.shape[1] != num_classes:
                    raise ValueError("Model output does not match num_classes")
                loss_sum += nn.functional.cross_entropy(
                    logits, targets, reduction="sum"
                ).item()
                predictions = logits.argmax(dim=1)
                indices = targets.cpu() * num_classes + predictions.cpu()
                confusion += torch.bincount(
                    indices, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
                count += targets.shape[0]
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation batches are empty")
    true_positive = confusion.diag().float()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    per_class_f1 = 2 * true_positive / denominator.clamp_min(1)
    return {
        "loss": loss_sum / count,
        "accuracy": true_positive.sum().item() / count,
        "macro_f1": per_class_f1.mean().item(),
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion,
    }
