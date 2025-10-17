"""
Reward Function Exploration Framework
Enables agent to explore optimal reward function configurations for different personalities
Uses multi-armed bandit and Bayesian optimization approaches
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .personality_assessment import BigFiveProfile


@dataclass
class RewardWeightConfig:
    """Reward function weight configuration"""
    explicit_feedback: float = 0.3
    behavioral: float = 0.25
    clinical: float = 0.25
    safety: float = 0.15
    engagement: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return {
            'explicit_feedback': self.explicit_feedback,
            'behavioral': self.behavioral,
            'clinical': self.clinical,
            'safety': self.safety,
            'engagement': self.engagement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RewardWeightConfig':
        return cls(
            explicit_feedback=data.get('explicit_feedback', 0.3),
            behavioral=data.get('behavioral', 0.25),
            clinical=data.get('clinical', 0.25),
            safety=data.get('safety', 0.15),
            engagement=data.get('engagement', 0.05),
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization"""
        return np.array([
            self.explicit_feedback,
            self.behavioral,
            self.clinical,
            self.safety,
            self.engagement,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RewardWeightConfig':
        """Create from numpy array"""
        return cls(
            explicit_feedback=float(arr[0]),
            behavioral=float(arr[1]),
            clinical=float(arr[2]),
            safety=float(arr[3]),
            engagement=float(arr[4]),
        )

    def normalize(self) -> 'RewardWeightConfig':
        """Normalize weights to sum to 1.0"""
        arr = self.to_array()
        arr = np.abs(arr)  # Ensure positive
        total = arr.sum()
        if total > 0:
            arr = arr / total
        return RewardWeightConfig.from_array(arr)


@dataclass
class ExplorationResult:
    """Result of a reward configuration exploration"""
    personality_profile: BigFiveProfile
    weight_config: RewardWeightConfig
    performance_metrics: Dict[str, float]  # user_satisfaction, engagement, clinical_improvement
    sample_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_overall_score(self) -> float:
        """Calculate overall performance score"""
        # Weighted combination of metrics
        return (
            0.5 * self.performance_metrics.get('user_satisfaction', 0.0) +
            0.3 * self.performance_metrics.get('clinical_improvement', 0.0) +
            0.2 * self.performance_metrics.get('engagement', 0.0)
        )


class PersonalityCluster:
    """
    Cluster similar personality profiles for efficient exploration
    Uses k-means or hierarchical clustering on Big Five dimensions
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.cluster_centers: List[np.ndarray] = []
        self.cluster_assignments: Dict[str, int] = {}  # user_id -> cluster_id

    def fit(self, profiles: List[Tuple[str, BigFiveProfile]]):
        """
        Cluster personality profiles

        Args:
            profiles: List of (user_id, BigFiveProfile) tuples
        """
        if len(profiles) < self.n_clusters:
            # Not enough data, each profile is its own cluster
            self.cluster_centers = [self._profile_to_vector(p) for _, p in profiles]
            return

        # Convert profiles to vectors
        vectors = np.array([self._profile_to_vector(p) for _, p in profiles])

        # Simple k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)

        self.cluster_centers = kmeans.cluster_centers_

        for (user_id, _), label in zip(profiles, labels):
            self.cluster_assignments[user_id] = int(label)

    def predict_cluster(self, profile: BigFiveProfile) -> int:
        """Assign a new profile to nearest cluster"""
        if len(self.cluster_centers) == 0:
            return 0

        vector = self._profile_to_vector(profile)
        distances = [np.linalg.norm(vector - center) for center in self.cluster_centers]
        return int(np.argmin(distances))

    @staticmethod
    def _profile_to_vector(profile: BigFiveProfile) -> np.ndarray:
        """Convert personality profile to vector for clustering"""
        return np.array([
            profile.openness,
            profile.conscientiousness,
            profile.extraversion,
            profile.agreeableness,
            profile.neuroticism,
        ])


class RewardExplorer:
    """
    Explores optimal reward function configurations for different personalities
    Uses contextual multi-armed bandit approach
    """

    def __init__(
        self,
        storage_dir: str = "psychology_agent/data/reward_exploration",
        exploration_rate: float = 0.2,  # Epsilon for epsilon-greedy
        learning_rate: float = 0.1,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate

        # Cluster personality types
        self.personality_clusters = PersonalityCluster(n_clusters=5)

        # Store exploration results per cluster
        # cluster_id -> list of (config, performance_score, sample_count)
        self.cluster_configs: Dict[int, List[Tuple[RewardWeightConfig, float, int]]] = defaultdict(list)

        # Store all exploration history
        self.exploration_history: List[ExplorationResult] = []

        # Load previous exploration data
        self._load_exploration_data()

    def select_reward_config(
        self,
        personality_profile: BigFiveProfile,
        exploration_mode: bool = True
    ) -> RewardWeightConfig:
        """
        Select reward configuration for a given personality

        Args:
            personality_profile: User's personality profile
            exploration_mode: If True, use epsilon-greedy; if False, always exploit best

        Returns:
            RewardWeightConfig to use
        """
        # Assign to cluster
        cluster_id = self.personality_clusters.predict_cluster(personality_profile)

        # Epsilon-greedy selection
        if exploration_mode and np.random.random() < self.exploration_rate:
            # EXPLORE: Try a random configuration
            config = self._generate_random_config()
            print(f"🔍 Exploring new reward config for cluster {cluster_id}")
        else:
            # EXPLOIT: Use best known configuration for this cluster
            config = self._get_best_config_for_cluster(cluster_id)
            print(f"✅ Using best known reward config for cluster {cluster_id}")

        return config

    def update_performance(
        self,
        personality_profile: BigFiveProfile,
        weight_config: RewardWeightConfig,
        performance_metrics: Dict[str, float]
    ):
        """
        Update performance data after using a reward configuration

        Args:
            personality_profile: User's personality
            weight_config: The reward config that was used
            performance_metrics: Observed performance (satisfaction, engagement, etc.)
        """
        cluster_id = self.personality_clusters.predict_cluster(personality_profile)

        # Create exploration result
        result = ExplorationResult(
            personality_profile=personality_profile,
            weight_config=weight_config,
            performance_metrics=performance_metrics,
            sample_count=1,
            timestamp=datetime.now()
        )

        self.exploration_history.append(result)

        # Update cluster config performance using exponential moving average
        overall_score = result.get_overall_score()
        self._update_cluster_config(cluster_id, weight_config, overall_score)

        # Periodically save
        if len(self.exploration_history) % 10 == 0:
            self._save_exploration_data()

        print(f"📊 Updated performance for cluster {cluster_id}: score={overall_score:.3f}")

    def _generate_random_config(self) -> RewardWeightConfig:
        """Generate a random reward weight configuration"""
        # Sample from Dirichlet distribution to ensure weights sum to 1
        weights = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0])
        return RewardWeightConfig.from_array(weights)

    def _get_best_config_for_cluster(self, cluster_id: int) -> RewardWeightConfig:
        """Get the best performing configuration for a cluster"""
        if cluster_id not in self.cluster_configs or not self.cluster_configs[cluster_id]:
            # No data yet, return default config
            return RewardWeightConfig()

        # Find config with highest score
        best_config, best_score, _ = max(
            self.cluster_configs[cluster_id],
            key=lambda x: x[1]  # Sort by score
        )

        return best_config

    def _update_cluster_config(
        self,
        cluster_id: int,
        weight_config: RewardWeightConfig,
        performance_score: float
    ):
        """Update or add configuration performance for a cluster"""
        # Check if this config already exists (within tolerance)
        config_array = weight_config.to_array()

        for i, (existing_config, existing_score, sample_count) in enumerate(self.cluster_configs[cluster_id]):
            existing_array = existing_config.to_array()

            # If configs are similar (L2 distance < threshold)
            if np.linalg.norm(config_array - existing_array) < 0.1:
                # Update existing config with exponential moving average
                new_score = (
                    existing_score * (sample_count / (sample_count + 1)) +
                    performance_score * (1 / (sample_count + 1))
                )
                self.cluster_configs[cluster_id][i] = (
                    existing_config,
                    new_score,
                    sample_count + 1
                )
                return

        # New config, add to cluster
        self.cluster_configs[cluster_id].append((weight_config, performance_score, 1))

        # Keep only top K configs per cluster to prevent memory growth
        if len(self.cluster_configs[cluster_id]) > 20:
            self.cluster_configs[cluster_id] = sorted(
                self.cluster_configs[cluster_id],
                key=lambda x: x[1],
                reverse=True
            )[:20]

    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get statistics about exploration progress"""
        stats = {
            'total_explorations': len(self.exploration_history),
            'n_clusters': len(self.cluster_configs),
            'configs_per_cluster': {
                cluster_id: len(configs)
                for cluster_id, configs in self.cluster_configs.items()
            },
        }

        if self.exploration_history:
            recent_results = self.exploration_history[-100:]  # Last 100
            stats['recent_avg_score'] = np.mean([
                r.get_overall_score() for r in recent_results
            ])

        return stats

    def visualize_exploration(self) -> str:
        """Generate a text summary of exploration results"""
        stats = self.get_exploration_statistics()

        summary = f"""
Reward Function Exploration Summary
====================================
Total Explorations: {stats['total_explorations']}
Number of Personality Clusters: {stats['n_clusters']}

Configurations per Cluster:
"""
        for cluster_id, n_configs in stats.get('configs_per_cluster', {}).items():
            summary += f"  Cluster {cluster_id}: {n_configs} configurations tested\n"

        if 'recent_avg_score' in stats:
            summary += f"\nRecent Average Score: {stats['recent_avg_score']:.3f}\n"

        # Show best config for each cluster
        summary += "\nBest Configurations by Cluster:\n"
        for cluster_id in sorted(self.cluster_configs.keys()):
            best_config = self._get_best_config_for_cluster(cluster_id)
            summary += f"\nCluster {cluster_id}:\n"
            for key, value in best_config.to_dict().items():
                summary += f"  {key}: {value:.3f}\n"

        return summary

    def _save_exploration_data(self):
        """Save exploration data to disk"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'exploration_history': [
                {
                    'personality': result.personality_profile.to_dict(),
                    'weights': result.weight_config.to_dict(),
                    'metrics': result.performance_metrics,
                    'score': result.get_overall_score(),
                    'timestamp': result.timestamp.isoformat(),
                }
                for result in self.exploration_history[-1000:]  # Keep last 1000
            ],
            'cluster_configs': {
                str(cluster_id): [
                    {
                        'weights': config.to_dict(),
                        'score': score,
                        'samples': samples,
                    }
                    for config, score, samples in configs
                ]
                for cluster_id, configs in self.cluster_configs.items()
            },
        }

        file_path = self.storage_dir / "exploration_data.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_exploration_data(self):
        """Load previous exploration data"""
        file_path = self.storage_dir / "exploration_data.json"

        if not file_path.exists():
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Restore cluster configs
            for cluster_id_str, configs in data.get('cluster_configs', {}).items():
                cluster_id = int(cluster_id_str)
                self.cluster_configs[cluster_id] = [
                    (
                        RewardWeightConfig.from_dict(c['weights']),
                        c['score'],
                        c['samples']
                    )
                    for c in configs
                ]

            print(f"✓ Loaded {len(data.get('exploration_history', []))} exploration records")

        except Exception as e:
            print(f"Failed to load exploration data: {e}")


# Global singleton
_reward_explorer = None


def get_reward_explorer() -> RewardExplorer:
    """Get global RewardExplorer instance"""
    global _reward_explorer
    if _reward_explorer is None:
        _reward_explorer = RewardExplorer()
    return _reward_explorer
