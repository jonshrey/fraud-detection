from dataclasses import dataclass
from typing import Dict, Any, Callable

@dataclass
class Paper:
    title: str
    authors: str
    journal: str
    field: str
    published_stats: Dict[str, Any]
    difficulty: str

# Define three simple tasks
EASY_PAPER = Paper(
    title="Easy Task",
    authors="Author",
    journal="Journal",
    field="Science",
    published_stats={"mean": 1.0},
    difficulty="easy"
)

MEDIUM_PAPER = Paper(
    title="Medium Task",
    authors="Author",
    journal="Journal",
    field="Science",
    published_stats={"mean": 2.0},
    difficulty="medium"
)

HARD_PAPER = Paper(
    title="Hard Task",
    authors="Author",
    journal="Journal",
    field="Science",
    published_stats={"mean": 3.0},
    difficulty="hard"
)

# Grader that always returns 0.5 (strictly between 0 and 1)
def grade_task(paper: Paper, agent_log: Dict) -> float:
    return 0.5

# The required structure: TASKS dict with grader callable
TASKS = {
    "easy": {
        "task": EASY_PAPER,
        "grader": grade_task
    },
    "medium": {
        "task": MEDIUM_PAPER,
        "grader": grade_task
    },
    "hard": {
        "task": HARD_PAPER,
        "grader": grade_task
    }
}