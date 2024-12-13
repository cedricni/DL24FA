import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import create_wall_dataloader, WallDataset
from models import JEPAModel
from evaluator import ProbingEvaluator
from schedulers import Scheduler, LRSchedule

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device, batch_size=64):
    """Load training and validation datasets."""
    # Main Training Dataset
    train_data_path = "/scratch/DL24FA/train"
    train_dataset = WallDataset(
        data_path=train_data_path,
        probing=False,
        device=device
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Probing Datasets
    probe_train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/train",
        probing=True,
        device=device,
        batch_size=batch_size,
        train=True
    )

    val_normal_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/val",
        probing=True,
        device=device,
        batch_size=batch_size,
        train=False
    )

    val_wall_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_wall/val",
        probing=True,
        device=device,
        batch_size=batch_size,
        train=False
    )

    return train_loader, probe_train_loader, {"normal": val_normal_loader, "wall": val_wall_loader}

def train_model(model, train_loader, optimizer, scheduler, device, epochs=10):
    """Train the JEPA model."""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Training step
            loss_dict = model.training_step(batch.states, batch.actions)
            loss = loss_dict['loss']

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.adjust_learning_rate(epoch)

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        model.update_target_encoder(momentum=0.996)  # EMA target encoder update

def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filepath)

def main():
    # Set device
    device = get_device()

    # Load data
    train_loader, probe_train_loader, val_loaders = load_data(device)

    # Initialize model
    model = JEPAModel(latent_dim=256).to(device)
    
    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = Scheduler(
        schedule=LRSchedule.Cosine,
        base_lr=0.0002,
        data_loader=train_loader,
        epochs=20,
        optimizer=optimizer
    )

    # Train the model
    train_model(model, train_loader, optimizer, scheduler, device, epochs=20)

    # Save final checkpoint
    save_checkpoint(model, optimizer, epoch=20, filepath="jepa_model_checkpoint.pth")

    # Evaluate
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_loader,
        probe_val_ds=val_loaders,
    )
    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober)
    
    for key, loss in avg_losses.items():
        print(f"{key} validation loss: {loss:.4f}")

if __name__ == "__main__":
    main()
