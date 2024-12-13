import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BoxDataset, get_transforms
from model import get_model
from utils import save_model, log_tensorboard
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter


def train_model(
    data_dir, num_epochs=10, batch_size=3, lr=1e-4, save_folder="../../weights"
):
    """
    Trains the Mask R-CNN model on the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        lr (flaot): Learning rate.
        save_folder (str): Directory to save model checkpoints.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = BoxDataset(data_dir, transforms=get_transforms(train=True))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    model = get_model(num_classes=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    writer = SummaryWriter()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        log_tensorboard(writer, epoch, epoch_loss)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_folder, f"{epoch+1}_model.pth")
            save_model(model, optimizer, epoch, save_path)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    data_dir = "../../Data"
    train_model(data_dir, num_epochs=200)
