#!/usr/bin/env python3
"""
Evaluation script for Semantic-STGCNN

This script provides comprehensive evaluation of trained Semantic-STGCNN models,
including metric calculation, visualization, and result analysis.

Usage:
    python -m semantic_stgcnn.scripts.evaluate --checkpoint checkpoints/best.pth --data_dir ./DATASET/SDD_WITH_FEATURES_SPLITTED
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from semantic_stgcnn.models import SemanticSTGCNN
from semantic_stgcnn.utils import TrajectoryDataset, ade, fde, calculate_trajectory_metrics
from semantic_stgcnn.config import Config, load_config, get_evaluation_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('config', {}).get('model', {})
    
    # Create model
    model = SemanticSTGCNN(
        n_stgcnn=model_config.get('n_stgcnn', 2),
        n_txpcnn=model_config.get('n_txpcnn', 5),
        input_feat=model_config.get('input_feat', 21),
        output_feat=model_config.get('output_feat', 2),
        seq_len=model_config.get('seq_len', 8),
        pred_seq_len=model_config.get('pred_seq_len', 12),
        kernel_size=model_config.get('kernel_size', 3)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, checkpoint


def create_data_loader(config: Config) -> DataLoader:
    """Create data loader for evaluation."""
    logger.info("Creating data loader...")
    
    dataset = TrajectoryDataset(
        data_dir=config.dataset.data_dir,
        obs_len=config.dataset.obs_len,
        pred_len=config.dataset.pred_len,
        skip=config.dataset.skip,
        threshold=config.dataset.threshold,
        min_ped=config.dataset.min_ped,
        delim=config.dataset.delim,
        norm_lap_matr=config.dataset.norm_lap_matr
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=False
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    return data_loader


def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader, 
                  device: torch.device,
                  config: Config) -> Dict[str, Any]:
    """Evaluate model on dataset."""
    logger.info("Starting evaluation...")
    
    model.eval()
    
    # Storage for predictions and targets
    all_predictions = []
    all_targets = []
    all_observed = []
    all_counts = []
    
    # Detailed metrics storage
    batch_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating')):
            # Unpack batch data
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, V_obs, A_obs) = batch
            
            # Move to device
            V_obs = V_obs.to(device)
            A_obs = A_obs.to(device)
            pred_traj = pred_traj.to(device)
            obs_traj = obs_traj.to(device)
            
            # Forward pass
            V_pred = model(V_obs, A_obs)
            
            # Extract coordinates (first 2 dimensions)
            pred_coords = V_pred[:, :, :, :2]  # (batch, pred_len, num_peds, 2)
            target_coords = pred_traj[:, :, :, :2]
            obs_coords = obs_traj[:, :, :, :2]
            
            # Store for global metrics
            all_predictions.append(pred_coords.cpu().numpy())
            all_targets.append(target_coords.cpu().numpy())
            all_observed.append(obs_coords.cpu().numpy())
            all_counts.append(pred_coords.shape[2])  # Number of pedestrians
            
            # Calculate batch-level metrics
            batch_metric = calculate_trajectory_metrics(pred_coords, target_coords)
            batch_metric['batch_idx'] = batch_idx
            batch_metric['num_pedestrians'] = pred_coords.shape[2]
            batch_metrics.append(batch_metric)
    
    # Calculate global metrics
    logger.info("Calculating global metrics...")
    
    try:
        global_ade = ade(all_predictions, all_targets, all_counts)
        global_fde = fde(all_predictions, all_targets, all_counts)
    except Exception as e:
        logger.error(f"Error calculating ADE/FDE: {e}")
        global_ade = float('inf')
        global_fde = float('inf')
    
    # Aggregate batch metrics
    batch_df = pd.DataFrame(batch_metrics)
    
    results = {
        'global_metrics': {
            'ade': global_ade,
            'fde': global_fde,
            'num_sequences': len(all_predictions),
            'total_pedestrians': sum(all_counts)
        },
        'batch_metrics': batch_df,
        'predictions': all_predictions,
        'targets': all_targets,
        'observed': all_observed,
        'counts': all_counts
    }
    
    logger.info(f"Evaluation completed!")
    logger.info(f"Global ADE: {global_ade:.4f}")
    logger.info(f"Global FDE: {global_fde:.4f}")
    
    return results


def create_visualizations(results: Dict[str, Any], output_dir: Path) -> None:
    """Create evaluation visualizations."""
    logger.info("Creating visualizations...")
    
    # Create output directory
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    batch_metrics = results['batch_metrics']
    
    # ADE distribution
    axes[0, 0].hist(batch_metrics['avg_displacement_error'], bins=30, alpha=0.7)
    axes[0, 0].set_title('Average Displacement Error Distribution')
    axes[0, 0].set_xlabel('ADE')
    axes[0, 0].set_ylabel('Frequency')
    
    # FDE distribution
    axes[0, 1].hist(batch_metrics['final_displacement_error'], bins=30, alpha=0.7)
    axes[0, 1].set_title('Final Displacement Error Distribution')
    axes[0, 1].set_xlabel('FDE')
    axes[0, 1].set_ylabel('Frequency')
    
    # MSE distribution
    axes[1, 0].hist(batch_metrics['mse'], bins=30, alpha=0.7)
    axes[1, 0].set_title('Mean Squared Error Distribution')
    axes[1, 0].set_xlabel('MSE')
    axes[1, 0].set_ylabel('Frequency')
    
    # Number of pedestrians vs ADE
    axes[1, 1].scatter(batch_metrics['num_pedestrians'], batch_metrics['avg_displacement_error'], alpha=0.6)
    axes[1, 1].set_title('Number of Pedestrians vs ADE')
    axes[1, 1].set_xlabel('Number of Pedestrians')
    axes[1, 1].set_ylabel('ADE')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample trajectory visualizations
    plot_sample_trajectories(results, viz_dir, num_samples=5)
    
    logger.info(f"Visualizations saved to {viz_dir}")


def plot_sample_trajectories(results: Dict[str, Any], output_dir: Path, num_samples: int = 5) -> None:
    """Plot sample trajectory predictions."""
    predictions = results['predictions']
    targets = results['targets']
    observed = results['observed']
    
    # Select random samples
    sample_indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        pred = predictions[idx][0]  # First batch item
        target = targets[idx][0]
        obs = observed[idx][0]
        
        # Plot trajectories for each pedestrian
        num_peds = pred.shape[1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for ped in range(num_peds):
            # Observed trajectory
            obs_traj = obs[:, ped, :]
            ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'o-', 
                   label=f'Observed Ped {ped}', linewidth=2, markersize=4)
            
            # Ground truth future
            target_traj = target[:, ped, :]
            ax.plot(target_traj[:, 0], target_traj[:, 1], 's-', 
                   label=f'Ground Truth Ped {ped}', linewidth=2, markersize=4)
            
            # Predicted future
            pred_traj = pred[:, ped, :]
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], '^--', 
                   label=f'Predicted Ped {ped}', linewidth=2, markersize=4)
            
            # Connect observed to predicted
            ax.plot([obs_traj[-1, 0], pred_traj[0, 0]], 
                   [obs_traj[-1, 1], pred_traj[0, 1]], 
                   'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Sample Trajectory Prediction {i+1}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_trajectory_{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save evaluation results."""
    logger.info("Saving results...")
    
    # Save global metrics
    global_metrics = results['global_metrics']
    with open(output_dir / 'global_metrics.txt', 'w') as f:
        f.write("Semantic-STGCNN Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Average Displacement Error (ADE): {global_metrics['ade']:.4f}\n")
        f.write(f"Final Displacement Error (FDE): {global_metrics['fde']:.4f}\n")
        f.write(f"Number of sequences: {global_metrics['num_sequences']}\n")
        f.write(f"Total pedestrians: {global_metrics['total_pedestrians']}\n")
    
    # Save batch metrics
    batch_metrics = results['batch_metrics']
    batch_metrics.to_csv(output_dir / 'batch_metrics.csv', index=False)
    
    # Save summary statistics
    summary_stats = batch_metrics.describe()
    summary_stats.to_csv(output_dir / 'summary_statistics.csv')
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Semantic-STGCNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_evaluation_config()
    
    # Override with command line arguments
    config.dataset.data_dir = args.data_dir
    config.training.batch_size = args.batch_size
    
    # Set up device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Create data loader
    data_loader = create_data_loader(config)
    
    # Evaluate model
    results = evaluate_model(model, data_loader, device, config)
    
    # Save results
    save_results(results, output_dir)
    
    # Create visualizations
    if args.visualize:
        create_visualizations(results, output_dir)
    
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
