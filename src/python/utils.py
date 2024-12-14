import os
import torch


def save_model(model, optimizer, epoch, path="model.pth"):
    """
    Saves the model state to a specified file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        epoch (int): The current training epoch.
        path (str):Path to save the model state.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def log_tensorboard(writer, epoch, loss):
    """
    Logs training metrics to TensorBoard

    Args:
        writer (SummaryWriter): TensorBoard writer object.
        epoch (int): Current epoch.
        loss (float): Training loss for the epoch.
    """
    writer.add_scalar("Loss/Train", loss, epoch + 1)
