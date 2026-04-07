from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import numpy as np

@dataclass
class RawDataset:
    id: str
    description: str
    data: Dict[str, Any]
    fabrication_type: Optional[str] = None

@dataclass
class Paper:
    title: str
    authors: str
    journal: str
    field: str
    published_stats: Dict[str, Any]
    raw_datasets: List[RawDataset]
    ground_truth_fabrication: Dict[str, Any]
    author_explanations: Dict[str, str]
    difficulty: str

# ----- EASY -----
EASY_DATASETS = [
    RawDataset(
        id="raw_data_1",
        description="Raw measurements from experiment 1",
        data={"values": [1.23, 1.24, 1.23, 1.24, 5.67, 5.67, 1.23, 1.24]},
        fabrication_type="duplicate_rows"
    )
]
EASY_PAPER = Paper(
    title="The Effect of Temperature on Enzyme Activity",
    authors="Smith, J.",
    journal="Journal of Biochemistry",
    field="biochemistry",
    published_stats={"mean": 2.45, "sd": 1.23, "n": 8},
    raw_datasets=EASY_DATASETS,
    ground_truth_fabrication={"type": "duplicate_rows", "location": "raw_data_1", "severity": 3},
    author_explanations={"raw_data_1": "These are the correct measurements. No duplicates exist."},
    difficulty="easy"
)

# ----- MEDIUM -----
def generate_benford_violating_data(n=100):
    np.random.seed(42)
    first_digit = np.random.choice(range(1,10), size=n, p=np.log10(1+1/np.arange(1,10)))
    last_digit = np.random.randint(0,10, size=n)
    numbers = first_digit * 10 + last_digit
    return numbers.tolist()

MEDIUM_DATASETS = [
    RawDataset(
        id="raw_data_2",
        description="Reaction time measurements (ms)",
        data={"reaction_times": generate_benford_violating_data(100)},
        fabrication_type="benford_violation"
    )
]
MEDIUM_PAPER = Paper(
    title="Quantum Dot Catalysis: Reaction Kinetics Study",
    authors="Chen, L., Patel, R.",
    journal="Nano Letters",
    field="nanotechnology",
    published_stats={"mean": 45.2, "sd": 12.3, "n": 100},
    raw_datasets=MEDIUM_DATASETS,
    ground_truth_fabrication={"type": "benford_violation", "location": "raw_data_2", "severity": 4},
    author_explanations={"raw_data_2": "The data was collected with high precision. All measurements are authentic."},
    difficulty="medium"
)

# ----- HARD -----
HARD_DATASET_A = RawDataset(
    id="exp_A",
    description="Experiment A: catalyst efficiency measurements",
    data={
        "efficiency": [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84, 0.87],
        "temperature": [298, 299, 297, 298, 298, 299, 297, 298],
        "timestamps": ["2023-01-01 10:00", "2023-01-01 10:05", "2023-01-01 10:10", "2023-01-01 10:15",
                       "2023-01-01 10:20", "2023-01-01 10:25", "2023-01-01 10:30", "2023-01-01 10:35"]
    },
    fabrication_type="impossible_correlation"
)
HARD_DATASET_B = RawDataset(
    id="exp_B",
    description="Experiment B: same catalyst, different lab",
    data={
        "efficiency": [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84, 0.87],
        "temperature": [298, 299, 297, 298, 298, 299, 297, 298],
        "timestamps": ["2023-01-01 10:00", "2023-01-01 10:05", "2023-01-01 10:10", "2023-01-01 10:15",
                       "2023-01-01 10:20", "2023-01-01 10:25", "2023-01-01 10:30", "2023-01-01 10:35"]
    },
    fabrication_type="timestamp_reuse"
)
HARD_PAPER = Paper(
    title="Breakthrough Catalyst for CO2 Reduction",
    authors="Williams, T., Garcia, M., Lee, S.",
    journal="Nature Catalysis",
    field="chemistry",
    published_stats={
        "mean_efficiency_A": 0.855, "sd_A": 0.012, "n_A": 8,
        "mean_efficiency_B": 0.855, "sd_B": 0.012, "n_B": 8,
        "correlation_AB": 0.999
    },
    raw_datasets=[HARD_DATASET_A, HARD_DATASET_B],
    ground_truth_fabrication={"type": "multi_paper_fabrication", "location": "both datasets", "severity": 5},
    author_explanations={
        "exp_A": "Data collected at UC Berkeley.",
        "exp_B": "Data collected at MIT independently. We stand by our results."
    },
    difficulty="hard"
)

PAPERS = {
    "easy": EASY_PAPER,
    "medium": MEDIUM_PAPER,
    "hard": HARD_PAPER,
}


class Grader:
    @staticmethod
    def grade(paper: Paper, agent_log: Dict) -> float:
        """
        Score the agent's fraud detection attempt.
        Returns a float strictly in (0, 1).
        """
        try:
            gt = paper.ground_truth_fabrication
            gt_type = gt.get("type", "")
            gt_location = gt.get("location", "")
            severity = gt.get("severity", 3)

            detected_type = agent_log.get("fabrication_type", "")
            detected_location = agent_log.get("location", "")
            confidence = float(agent_log.get("confidence", 0.5))

            # Clamp confidence to (0, 1) open interval
            confidence = max(0.01, min(0.99, confidence))

            score = 0.1  # baseline: always strictly > 0

            # Reward correct type detection
            if detected_type and detected_type.lower() == gt_type.lower():
                score += 0.4

            # Reward correct location detection
            if detected_location and gt_location.lower() in detected_location.lower():
                score += 0.3

            # Blend in agent's own confidence as a small factor
            score += 0.1 * confidence

            # Normalize by severity (harder cases worth more, but keep in range)
            score = score * (0.8 + 0.04 * severity)

            # Strictly clamp to (0, 1) open interval — never touch 0.0 or 1.0
            score = max(0.01, min(0.99, score))
            return float(score)

        except Exception:
            # Safe fallback that is strictly within (0, 1)
            return 0.5


TASKS = {
    name: {"task": paper, "grader": Grader.grade}
    for name, paper in PAPERS.items()
}