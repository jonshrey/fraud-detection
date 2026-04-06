from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, List

class Observation(BaseModel):
    paper_metadata: Dict[str, Any]
    published_stats: Dict[str, Any]
    available_raw_data: List[str]
    test_results: Dict[str, Any]
    author_responses: List[str]
    step_count: int

class Action(BaseModel):
    action_type: Literal["request_raw_data", "run_statistical_test", "request_author_explanation", "flag_paper", "issue_verdict"]
    dataset_id: Optional[str] = None
    test_name: Optional[Literal["benford", "digit_frequency", "outlier_detection", "correlation_check", "timestamp_consistency"]] = None
    severity: Optional[int] = None
    verdict: Optional[Literal["retract", "require_revision", "accept"]] = None

class Reward(BaseModel):
    value: float
    reasons: List[str]