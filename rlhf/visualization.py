"""
Visualization Tools for Personality-Reward Exploration
Generates visualizations and reports for analyzing exploration results
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be text-only.")


class ExplorationVisualizer:
    """
    Visualizes personality-reward exploration results
    """

    def __init__(self, exploration_data_path: str = "psychology_agent/data/reward_exploration/exploration_data.json"):
        self.data_path = Path(exploration_data_path)
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load exploration data"""
        if not self.data_path.exists():
            return {'exploration_history': [], 'cluster_configs': {}}

        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_text_report(self) -> str:
        """Generate comprehensive text report of exploration results"""

        report = []
        report.append("="*80)
        report.append(" "*20 + "PERSONALITY-REWARD EXPLORATION REPORT")
        report.append("="*80)
        report.append("")

        # Overview
        history = self.data.get('exploration_history', [])
        clusters = self.data.get('cluster_configs', {})

        report.append(f"Total Explorations: {len(history)}")
        report.append(f"Number of Personality Clusters: {len(clusters)}")
        report.append("")

        # Performance statistics
        if history:
            scores = [h['score'] for h in history]
            report.append("Overall Performance:")
            report.append(f"  Mean Score: {np.mean(scores):.3f}")
            report.append(f"  Std Dev:    {np.std(scores):.3f}")
            report.append(f"  Min Score:  {min(scores):.3f}")
            report.append(f"  Max Score:  {max(scores):.3f}")
            report.append("")

            # Learning trend
            if len(history) >= 20:
                early_scores = [h['score'] for h in history[:len(history)//4]]
                late_scores = [h['score'] for h in history[-len(history)//4:]]

                report.append("Learning Trend:")
                report.append(f"  Early Average (first 25%): {np.mean(early_scores):.3f}")
                report.append(f"  Late Average (last 25%):   {np.mean(late_scores):.3f}")
                improvement = ((np.mean(late_scores) - np.mean(early_scores)) / abs(np.mean(early_scores))) * 100
                report.append(f"  Improvement: {improvement:+.1f}%")
                report.append("")

        # Cluster-specific results
        report.append("-"*80)
        report.append("Best Configurations by Personality Cluster:")
        report.append("-"*80)
        report.append("")

        for cluster_id, configs in clusters.items():
            if not configs:
                continue

            # Find best config
            best_config = max(configs, key=lambda x: x['score'])

            report.append(f"Cluster {cluster_id}:")
            report.append(f"  Tested Configurations: {len(configs)}")
            report.append(f"  Best Score: {best_config['score']:.3f}")
            report.append(f"  Best Weights:")

            for component, weight in best_config['weights'].items():
                report.append(f"    {component:20s}: {weight:.3f}")

            report.append("")

        # Personality patterns
        if history:
            report.append("-"*80)
            report.append("Personality Trait Correlations:")
            report.append("-"*80)
            report.append("")

            report.append(self._analyze_trait_correlations(history))

        report.append("="*80)

        return "\n".join(report)

    def _analyze_trait_correlations(self, history: List[Dict]) -> str:
        """Analyze correlations between personality traits and optimal reward weights"""

        # Group by personality traits (high/low)
        trait_groups = {
            'high_neuroticism': [],
            'low_neuroticism': [],
            'high_extraversion': [],
            'low_extraversion': [],
        }

        for entry in history:
            personality = entry['personality']

            if personality['neuroticism'] > 0.6:
                trait_groups['high_neuroticism'].append(entry)
            else:
                trait_groups['low_neuroticism'].append(entry)

            if personality['extraversion'] > 0.6:
                trait_groups['high_extraversion'].append(entry)
            else:
                trait_groups['low_extraversion'].append(entry)

        lines = []

        for trait_name, entries in trait_groups.items():
            if not entries:
                continue

            # Average weights for this group
            avg_weights = defaultdict(float)
            for entry in entries:
                for component, weight in entry['weights'].items():
                    avg_weights[component] += weight

            n = len(entries)
            avg_weights = {k: v/n for k, v in avg_weights.items()}

            lines.append(f"{trait_name.replace('_', ' ').title()} (n={n}):")
            for component, weight in sorted(avg_weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {component:20s}: {weight:.3f}")
            lines.append("")

        return "\n".join(lines)

    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves over time"""

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate plots.")
            return

        history = self.data.get('exploration_history', [])
        if not history:
            print("No exploration history to plot")
            return

        # Extract scores over time
        scores = [h['score'] for h in history]
        timestamps = [datetime.fromisoformat(h['timestamp']) for h in history]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Raw scores over time
        axes[0].plot(scores, alpha=0.5, label='Raw Score')

        # Add moving average
        window = min(10, len(scores) // 10)
        if window > 1:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(scores)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')

        axes[0].set_xlabel('Interaction Number')
        axes[0].set_ylabel('Reward Score')
        axes[0].set_title('Learning Curve: Reward Score Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Score distribution by cluster
        cluster_scores = defaultdict(list)
        for entry in history:
            # Determine cluster (simplified - just use neuroticism for demo)
            neuroticism = entry['personality']['neuroticism']
            if neuroticism > 0.7:
                cluster = 'High Neuroticism'
            elif neuroticism < 0.3:
                cluster = 'Low Neuroticism'
            else:
                cluster = 'Moderate Neuroticism'

            cluster_scores[cluster].append(entry['score'])

        # Box plot
        axes[1].boxplot(
            [scores for scores in cluster_scores.values()],
            labels=list(cluster_scores.keys())
        )
        axes[1].set_ylabel('Reward Score')
        axes[1].set_title('Score Distribution by Personality Cluster')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        else:
            plt.show()

    def plot_weight_heatmap(self, save_path: Optional[str] = None):
        """Plot heatmap of reward weights by personality cluster"""

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate plots.")
            return

        clusters = self.data.get('cluster_configs', {})
        if not clusters:
            print("No cluster data to plot")
            return

        # Get best config for each cluster
        cluster_weights = {}
        for cluster_id, configs in clusters.items():
            if configs:
                best_config = max(configs, key=lambda x: x['score'])
                cluster_weights[f"Cluster {cluster_id}"] = best_config['weights']

        if not cluster_weights:
            return

        # Convert to matrix
        components = list(next(iter(cluster_weights.values())).keys())
        cluster_names = list(cluster_weights.keys())

        matrix = np.array([
            [cluster_weights[cluster][component] for component in components]
            for cluster in cluster_names
        ])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(components)))
        ax.set_yticks(np.arange(len(cluster_names)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.set_yticklabels(cluster_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight', rotation=270, labelpad=15)

        # Add values
        for i in range(len(cluster_names)):
            for j in range(len(components)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        ax.set_title('Optimal Reward Weights by Personality Cluster')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved heatmap to {save_path}")
        else:
            plt.show()

    def export_summary_csv(self, save_path: str = "exploration_summary.csv"):
        """Export exploration summary to CSV"""

        history = self.data.get('exploration_history', [])
        if not history:
            print("No data to export")
            return

        import csv

        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'score',
                'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
                'weight_explicit', 'weight_behavioral', 'weight_clinical', 'weight_safety', 'weight_engagement'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in history:
                row = {
                    'timestamp': entry['timestamp'],
                    'score': entry['score'],
                    'openness': entry['personality']['openness'],
                    'conscientiousness': entry['personality']['conscientiousness'],
                    'extraversion': entry['personality']['extraversion'],
                    'agreeableness': entry['personality']['agreeableness'],
                    'neuroticism': entry['personality']['neuroticism'],
                    'weight_explicit': entry['weights']['explicit_feedback'],
                    'weight_behavioral': entry['weights']['behavioral'],
                    'weight_clinical': entry['weights']['clinical'],
                    'weight_safety': entry['weights']['safety'],
                    'weight_engagement': entry['weights']['engagement'],
                }
                writer.writerow(row)

        print(f"✓ Exported data to {save_path}")


def generate_full_report(output_dir: str = "psychology_agent/reports"):
    """Generate complete visualization report"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = ExplorationVisualizer()

    # Text report
    report = visualizer.generate_text_report()
    report_file = output_path / f"exploration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Generated text report: {report_file}")

    # CSV export
    csv_file = output_path / f"exploration_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    visualizer.export_summary_csv(str(csv_file))

    # Plots
    if MATPLOTLIB_AVAILABLE:
        learning_curve_file = output_path / f"learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        visualizer.plot_learning_curves(str(learning_curve_file))

        heatmap_file = output_path / f"weight_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        visualizer.plot_weight_heatmap(str(heatmap_file))

    # Print report to console
    print("\n" + report)


if __name__ == "__main__":
    generate_full_report()
