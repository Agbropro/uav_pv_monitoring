from pathlib import Path
import torch

class Trainer:
    def __init__(self,model,optimizer,criterion, device, save_directory:str= "weights/mobilenet_v3_small"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(
            parents = True,
            exist_ok = True
        )
    def train_epoch(self,dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            running_loss += (
                loss.item() * images.size(0)
            )
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        return epoch_loss, epoch_accuracy
    
    def validate(self,dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.inference_mode(): # inference mode remove gradient calculation because validation does not need gradient tracking for loss propagation, thus more efficient in memory, torch default includes gradient tracking (no loss.backward())
            for images, labels in dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy
    
    def _save_checkpoint(self,path:Path,epoch: int, train_loss: float, val_loss: float, class_names: list[str] | None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "class_names": class_names,
        }

        torch.save(
            checkpoint,
            path,
        )
    def resume_from_last(self, class_names: list[str]) -> tuple[int, float]:
        last_path = self.save_directory / "last.pt"
        if not last_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {last_path}")

        checkpoint = torch.load(last_path, map_location=self.device, weights_only=True)
        if checkpoint["class_names"] != class_names:
            raise ValueError("Checkpoint classes differ from dataset classes")

        best_val_loss = checkpoint["val_loss"]
        best_path = self.save_directory / "best.pt"
        if best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
            if best_checkpoint["class_names"] != class_names:
                raise ValueError("Best checkpoint classes differ from dataset classes")
            best_val_loss = min(best_val_loss, best_checkpoint["val_loss"])

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"] + 1, best_val_loss

    def fit(self,train_loader,validation_loader,epochs: int,class_names: list[str] | None = None, resume: bool = False):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        best_val_loss = float("inf")
        start_epoch = 1
        if resume:
            if class_names is None:
                raise ValueError("class_names are required to resume training")
            start_epoch, best_val_loss = self.resume_from_last(class_names)
            if start_epoch > epochs:
                raise ValueError(
                    f"Checkpoint has completed epoch {start_epoch - 1}; "
                    f"increase epochs above that value to continue"
                )
            print(f"Resuming from epoch {start_epoch}")
        
        for epoch in range(start_epoch, epochs + 1):
            train_loss,train_accuracy = (self.train_epoch(train_loader))
            val_loss,val_accuracy = (self.validate(validation_loader))
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            
            print(
                f"Epoch {epoch} / {epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )
            
            self._save_checkpoint(
                path=self.save_directory / "last.pt",
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                class_names=class_names,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                self._save_checkpoint(
                    path= self.save_directory / "best.pt",
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    class_names = class_names
                )
                
                print("Saved New Best Model")
        return history
