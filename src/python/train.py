import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from utils import save_model, log_tensorboard
from dataset import BoxDataset, get_transforms
from model import get_model


def train_model(
    data_dir,
    num_epochs=30,
    batch_size=4,
    lr=1e-4,
    save_epochs=5,
    save_folder="../../weights",
):
    """
    Trains the Mask R-CNN model on the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        lr (flaot): Learning rate.
        save_epochs (int): Number of epochs to save the model.
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

    # Two classes: background (0) and upper_face (1)
    model = get_model(num_classes=2)
    model.to(device)

    # Optimizer commonly used to fine-tune Mask R-CNN
    optimizer = AdamW(model.parameters(), lr=lr)

    # ChatGPT was used to develop the logging code
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

        # Log metrics to TensorBoard
        log_tensorboard(writer, epoch, epoch_loss)

        # Print summary for the epoch
        print(f"Epoch {epoch+1}/{num_epochs} | " f"Loss: {epoch_loss:.4f}")

        # Save the model every "save_epochs" epochs
        if (epoch + 1) % save_epochs == 0:
            save_path = os.path.join(save_folder, f"{epoch+1}_model.pth")
            save_model(model, optimizer, epoch, save_path)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    data_dir = "../../Data"
    save_folder = "../../weights"

    # Start with 30 epochs and evaluate performance
    num_epochs = 10000
    # Change depending on GPU memory (this is good for around 8 GB)
    batch_size = 4
    # Good value for fine-tuning the pre-trained backbone
    lr = 1e-4
    # Save the model when this number of epochs has been completed
    save_epochs = 5

    train_model(data_dir, num_epochs, batch_size, lr, save_epochs, save_folder)
