#!/usr/bin/env python3
"""
Simple standalone evaluation script for trajectory predictions.
Calculates ADE (Average Displacement Error) and FDE (Final Displacement Error).
"""
import json
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def parse_planning_output(text):
    """Extract waypoints from planning output text."""
    # Look for pattern like [x, y] in the planning output
    pattern = r'\[([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\]'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    waypoints = [(float(x), float(y)) for x, y in matches]
    return np.array(waypoints)


def calculate_metrics(pred_waypoints, gt_waypoints):
    """Calculate ADE and FDE metrics."""
    if pred_waypoints is None or gt_waypoints is None:
        return None, None

    # Ensure same length (use minimum length)
    min_len = min(len(pred_waypoints), len(gt_waypoints))
    pred_waypoints = pred_waypoints[:min_len]
    gt_waypoints = gt_waypoints[:min_len]

    # Calculate displacement errors
    displacements = np.linalg.norm(pred_waypoints - gt_waypoints, axis=1)

    # ADE: Average Displacement Error
    ade = np.mean(displacements)

    # FDE: Final Displacement Error
    fde = displacements[-1] if len(displacements) > 0 else 0

    return ade, fde


def visualize_trajectories(pred_waypoints, gt_waypoints, sample_idx, output_dir):
    """Visualize predicted vs ground truth trajectories."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectories
    if pred_waypoints is not None and len(pred_waypoints) > 0:
        ax.plot(pred_waypoints[:, 0], pred_waypoints[:, 1], 'r-o',
                label='Predicted', linewidth=2, markersize=8)

    if gt_waypoints is not None and len(gt_waypoints) > 0:
        ax.plot(gt_waypoints[:, 0], gt_waypoints[:, 1], 'g-s',
                label='Ground Truth', linewidth=2, markersize=8)

    # Mark start and end points
    ax.plot(0, 0, 'ko', markersize=15, label='Start (Ego)')

    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters, forward)', fontsize=12)
    ax.set_ylabel('Y (meters, left)', fontsize=12)
    ax.set_title(f'Sample {sample_idx}: Trajectory Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.axis('equal')

    # Save figure
    if isinstance(sample_idx, int):
        output_path = Path(output_dir) / f'sample_{sample_idx:04d}.png'
    else:
        output_path = Path(output_dir) / f'sample_{sample_idx}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def evaluate_predictions(jsonl_file, output_file=None, viz_dir=None, viz_every=100):
    """Evaluate predictions from JSONL file."""
    print(f"Loading predictions from {jsonl_file}")

    predictions = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions")

    # Create visualization directory if needed
    if viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to {viz_dir}")

    # Collect metrics and sample data
    ades = []
    fdes = []
    valid_count = 0
    invalid_count = 0
    sample_data = []  # Store data for visualization

    for i, pred in enumerate(predictions):
        # Extract planning sections
        pred_text = pred.get('predict', '')
        gt_text = pred.get('label', '')

        # Extract <PLANNING>...</PLANNING> content
        pred_planning = re.search(r'<PLANNING>(.*?)</PLANNING>', pred_text, re.DOTALL)
        gt_planning = re.search(r'<PLANNING>(.*?)</PLANNING>', gt_text, re.DOTALL)

        if pred_planning and gt_planning:
            pred_waypoints = parse_planning_output(pred_planning.group(1))
            gt_waypoints = parse_planning_output(gt_planning.group(1))

            ade, fde = calculate_metrics(pred_waypoints, gt_waypoints)

            if ade is not None:
                ades.append(ade)
                fdes.append(fde)
                valid_count += 1

                # Store sample data
                sample_data.append({
                    'idx': i,
                    'pred_waypoints': pred_waypoints,
                    'gt_waypoints': gt_waypoints,
                    'ade': ade,
                    'fde': fde
                })

                # Visualize every N samples
                if viz_dir and (i % viz_every == 0 or i < 10):
                    visualize_trajectories(pred_waypoints, gt_waypoints, i, viz_dir)
            else:
                invalid_count += 1
        else:
            invalid_count += 1

    # Calculate overall metrics
    results = {
        'total_samples': len(predictions),
        'valid_samples': valid_count,
        'invalid_samples': invalid_count,
        'metrics': {
            'ADE_mean': float(np.mean(ades)) if ades else 0,
            'ADE_std': float(np.std(ades)) if ades else 0,
            'ADE_median': float(np.median(ades)) if ades else 0,
            'FDE_mean': float(np.mean(fdes)) if fdes else 0,
            'FDE_std': float(np.std(fdes)) if fdes else 0,
            'FDE_median': float(np.median(fdes)) if fdes else 0,
        }
    }

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Valid samples: {results['valid_samples']}")
    print(f"Invalid samples: {results['invalid_samples']}")
    print("\nMetrics:")
    print(f"  ADE (Average Displacement Error):")
    print(f"    Mean:   {results['metrics']['ADE_mean']:.3f} m")
    print(f"    Median: {results['metrics']['ADE_median']:.3f} m")
    print(f"    Std:    {results['metrics']['ADE_std']:.3f} m")
    print(f"  FDE (Final Displacement Error):")
    print(f"    Mean:   {results['metrics']['FDE_mean']:.3f} m")
    print(f"    Median: {results['metrics']['FDE_median']:.3f} m")
    print(f"    Std:    {results['metrics']['FDE_std']:.3f} m")
    print("="*60)

    # Save results if output file specified
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")

    # Visualize best and worst samples
    if viz_dir and sample_data:
        print(f"\nGenerating best/worst sample visualizations...")

        # Sort by ADE
        sorted_samples = sorted(sample_data, key=lambda x: x['ade'])

        # Best 5 samples
        for i, sample in enumerate(sorted_samples[:5]):
            visualize_trajectories(
                sample['pred_waypoints'],
                sample['gt_waypoints'],
                f"best_{i+1}_idx{sample['idx']}_ade{sample['ade']:.2f}",
                viz_dir
            )

        # Worst 5 samples
        for i, sample in enumerate(sorted_samples[-5:]):
            visualize_trajectories(
                sample['pred_waypoints'],
                sample['gt_waypoints'],
                f"worst_{i+1}_idx{sample['idx']}_ade{sample['ade']:.2f}",
                viz_dir
            )

        print(f"Visualizations saved to {viz_dir}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trajectory predictions')
    parser.add_argument('--jsonl_file', type=str, required=True,
                       help='Path to predictions JSONL file')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results (JSON)')
    parser.add_argument('--viz_dir', type=str, default=None,
                       help='Directory to save trajectory visualizations')
    parser.add_argument('--viz_every', type=int, default=100,
                       help='Visualize every N samples (default: 100)')

    args = parser.parse_args()

    evaluate_predictions(args.jsonl_file, args.output_file, args.viz_dir, args.viz_every)
