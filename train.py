"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import wandb
import torch
from torch import nn
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer

from torchvision import transforms


# Setup directories
train_dir = ""
test_dir = ""

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Define sweep configuration
sweep_config = {
    "method": "bayes",  # Use Bayesian optimization for efficient search
    "metric": {
        "name": "best_test_accuracy",  # This will track the best accuracy across all epochs
        "goal": "maximize",
    },
    "parameters": {
        "model_name": {"values": ["MoCo_CXR", "MoCo_CXR", "Chess"]},
        "learning_rate": {"values": [1e-5, 1e-4, 5e-4]},
        "weight_decay": {"values": [0.0, 1e-4, 1e-3]},
        "batch_size": {"value": 32},
        "epochs": {"value": 30},
        "optimizer": {"values": ["adam", "adamw"]},
    },
}


# 2. Define the training function for sweep
def train_sweep():
    """Training function that will be called by wandb agent"""

    # Initialize wandb run
    with wandb.init() as run:
        # Access sweep parameters via wandb.config
        config = wandb.config

        # Model selection based on config
        model_dict = {
            "DenseNet": create_DenseNet_chex,
            "MoCo_CXR": create_MoCo_CXR,
            "Chess": create_Chess,
        }

        # Create model
        model_fn = model_dict[config.model_name]
        model, model_transforms = model_fn()
        model = model.to(device)

        print(
            f"[INFO] Training {config.model_name} | LR: {config.learning_rate} | WD: {config.weight_decay}"
        )

        # Create optimizer based on config
        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:  # adamw
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Create dataloaders
        train_dataloader, test_dataloader = create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=model_transforms,
            batch_size=config.batch_size,
            split_ratio=0.8,
        )

        # Training
        start_time = timer()

        train_results = train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            epochs=config.epochs,
            device=device,
            writer=None,
            wandb=run,  # Pass the run object to your train function
        )

        end_time = timer()
        training_time = end_time - start_time

        # Log final metrics and summary
        final_test_acc = train_results["test_acc"][-1]
        final_test_loss = train_results["test_loss"][-1]
        best_test_acc = max(train_results["test_acc"])
        best_test_loss = min(train_results["test_loss"])

        run.log(
            {
                "total_training_time": training_time,
                "final_test_accuracy": final_test_acc,
                "final_test_loss": final_test_loss,
                "best_test_accuracy": best_test_acc,
                "best_test_loss": best_test_loss,
            }
        )

        # Log summary for sweep comparison
        run.summary["best_test_accuracy"] = best_test_acc
        run.summary["best_test_loss"] = best_test_loss
        run.summary["final_test_accuracy"] = final_test_acc

        print(f"[INFO] Completed in {training_time:.3f} seconds")
        print(f"[INFO] Final test accuracy: {final_test_acc:.4f}")
        print(f"[INFO] Best test accuracy: {best_test_acc:.4f}")
