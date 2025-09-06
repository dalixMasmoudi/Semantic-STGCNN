#!/usr/bin/env python3
"""
Training script for Semantic-STGCNN

This script provides a complete training pipeline for the Semantic-STGCNN model,
including data loading, model initialization, training loop, validation, and checkpointing.

Usage:
    python -m semantic_stgcnn.scripts.train --config configs/train_config.yaml
    python -m semantic_stgcnn.scripts.train --data_dir ./DATASET/SDD_WITH_FEATURES_SPLITTED
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from semantic_stgcnn.models import SemanticSTGCNN
from semantic_stgcnn.utils import TrajectoryDataset, bivariate_loss, ade, fde
from semantic_stgcnn.config import Config, load_config, get_default_config, add_config_args, validate_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'auto') -> torch.device:
    """Get the appropriate device for training."""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    return device


def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    logger.info("Creating data loaders...")
    
    # Training dataset
    train_dataset = TrajectoryDataset(
        data_dir=config.dataset.data_dir,
        obs_len=config.dataset.obs_len,
        pred_len=config.dataset.pred_len,
        skip=config.dataset.skip,
        threshold=config.dataset.threshold,
        min_ped=config.dataset.min_ped,
        delim=config.dataset.delim,
        norm_lap_matr=config.dataset.norm_lap_matr
    )
    
    # For now, use the same dataset for validation (in practice, use separate validation set)
    val_dataset = train_dataset
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.dataloader.shuffle_train,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.dataloader.shuffle_val,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=False
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_model(config: Config, device: torch.device) -> nn.Module:
    """Create and initialize the model."""
    logger.info("Creating model...")
    
    model = SemanticSTGCNN(
        n_stgcnn=config.model.n_stgcnn,
        n_txpcnn=config.model.n_txpcnn,
        input_feat=config.model.input_feat,
        output_feat=config.model.output_feat,
        seq_len=config.model.seq_len,
        pred_seq_len=config.model.pred_seq_len,
        kernel_size=config.model.kernel_size
    )
    
    model = model.to(device)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model


def create_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """Create optimizer."""
    if config.training.optimizer.type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps
        )
    elif config.training.optimizer.type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.training.optimizer.type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Config):
    """Create learning rate scheduler."""
    if config.training.scheduler.type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler.step_size,
            gamma=config.training.scheduler.gamma
        )
    elif config.training.scheduler.type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer,
                device: torch.device,
                config: Config,
                epoch: int,
                writer: SummaryWriter = None) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Unpack batch data
        (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, 
         non_linear_ped, loss_mask, V_obs, A_obs) = batch
        
        # Move to device
        V_obs = V_obs.to(device)
        A_obs = A_obs.to(device)
        pred_traj = pred_traj.to(device)
        non_linear_ped = non_linear_ped.to(device)
        obs_traj = obs_traj.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        V_pred = model(V_obs, A_obs)
        
        # Calculate loss
        loss = bivariate_loss(V_pred, pred_traj, non_linear_ped, obs_traj)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log to tensorboard
        if writer and batch_idx % config.logging.log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
    
    avg_loss = total_loss / num_batches
    return {'loss': avg_loss}


def validate_epoch(model: nn.Module,
                  val_loader: DataLoader,
                  device: torch.device,
                  config: Config,
                  epoch: int,
                  writer: SummaryWriter = None) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # For ADE/FDE calculation
    pred_all = []
    target_all = []
    count_all = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Unpack batch data
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, V_obs, A_obs) = batch
            
            # Move to device
            V_obs = V_obs.to(device)
            A_obs = A_obs.to(device)
            pred_traj = pred_traj.to(device)
            non_linear_ped = non_linear_ped.to(device)
            obs_traj = obs_traj.to(device)
            
            # Forward pass
            V_pred = model(V_obs, A_obs)
            
            # Calculate loss
            loss = bivariate_loss(V_pred, pred_traj, non_linear_ped, obs_traj)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions for ADE/FDE calculation
            pred_all.append(V_pred[:, :, :, :2].cpu().numpy())  # Only x,y coordinates
            target_all.append(pred_traj[:, :, :, :2].cpu().numpy())
            count_all.append(V_pred.shape[2])  # Number of pedestrians
    
    avg_loss = total_loss / num_batches
    
    # Calculate ADE and FDE
    try:
        avg_ade = ade(pred_all, target_all, count_all)
        avg_fde = fde(pred_all, target_all, count_all)
    except Exception as e:
        logger.warning(f"Error calculating ADE/FDE: {e}")
        avg_ade = float('inf')
        avg_fde = float('inf')
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/ADE', avg_ade, epoch)
        writer.add_scalar('Val/FDE', avg_fde, epoch)
    
    return {'loss': avg_loss, 'ade': avg_ade, 'fde': avg_fde}


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   scheduler,
                   epoch: int,
                   metrics: Dict[str, float],
                   config: Config,
                   is_best: bool = False) -> None:
    """Save model checkpoint."""
    checkpoint_dir = Path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.to_dict()
    }
    
    # Save latest checkpoint
    if config.output.save_last:
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
    
    # Save best checkpoint
    if is_best and config.output.save_best_only:
        torch.save(checkpoint, checkpoint_dir / 'best.pth')
        logger.info(f"Saved best model at epoch {epoch}")


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Semantic-STGCNN')
    parser = add_config_args(parser)
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            # Handle nested config keys
            if '.' in key:
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if not hasattr(current, k):
                        setattr(current, k, Config())
                    current = getattr(current, k)
                setattr(current, keys[-1], value)
            else:
                setattr(config, key, value)
    
    # Validate configuration
    validate_config(config)
    
    # Set up reproducibility
    set_seed(config.reproducibility.seed)
    
    # Create output directories
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up device
    device = get_device(config.hardware.device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Set up tensorboard logging
    writer = None
    if config.logging.tensorboard:
        writer = SummaryWriter(log_dir=config.logging.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch, writer)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, config, epoch, writer)
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch}/{config.training.num_epochs}")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
        
        # Check for best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, is_best)
        
        # Early stopping
        if patience_counter >= config.training.early_stopping.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("Training completed!")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
