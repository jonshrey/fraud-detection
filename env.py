from typing import Dict, Any, Optional, List
from models import Observation, Action, Reward
from tasks import PAPERS as TASKS, Paper, Grader, RawDataset
import copy
import numpy as np
from scipy import stats

class FraudDetectionEnv:
    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Task must be one of {list(TASKS.keys())}")
        self.task_name = task_name
        self.paper = TASKS[task_name]
        self.reset()
    
    def reset(self) -> Observation:
        self.step_count = 0
        self.done = False
        self.datasets_requested = []
        self.available_raw_data = []
        self.test_results = {}
        self.author_responses = []
        self.flags = []
        self.final_verdict = None
        self.reward_total = 0.0
        return self._get_observation()
    
    def _get_observation(self) -> Observation:
        available_desc = []
        for ds_id in self.available_raw_data:
            ds = next((d for d in self.paper.raw_datasets if d.id == ds_id), None)
            if ds:
                available_desc.append(f"{ds.id}: {ds.description} - first few rows: {str(ds.data)[:200]}")
        return Observation(
            paper_metadata={
                "title": self.paper.title,
                "authors": self.paper.authors,
                "journal": self.paper.journal,
                "field": self.paper.field
            },
            published_stats=self.paper.published_stats,
            available_raw_data=available_desc,
            test_results=self.test_results,
            author_responses=self.author_responses,
            step_count=self.step_count
        )
    
    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")
        
        self.step_count += 1
        reward_val = 0.0
        reward_reasons = []
        info = {}
        
        if action.action_type == "request_raw_data":
            if action.dataset_id:
                ds = next((d for d in self.paper.raw_datasets if d.id == action.dataset_id), None)
                if ds and action.dataset_id not in self.datasets_requested:
                    self.datasets_requested.append(action.dataset_id)
                    self.available_raw_data.append(action.dataset_id)
                    reward_val += 0.1
                    reward_reasons.append(f"Requested raw data: {action.dataset_id}")
                elif ds and action.dataset_id in self.datasets_requested:
                    reward_val -= 0.05
                    reward_reasons.append("Already requested that dataset")
                else:
                    reward_val -= 0.1
                    reward_reasons.append("Invalid dataset ID")
            else:
                reward_val -= 0.2
                reward_reasons.append("Missing dataset_id")
        
        elif action.action_type == "run_statistical_test":
            if action.test_name and self.available_raw_data:
                ds_id = self.available_raw_data[-1]
                ds = next(d for d in self.paper.raw_datasets if d.id == ds_id)
                result = self._run_test(action.test_name, ds)
                test_key = f"{ds_id}_{action.test_name}"
                if test_key not in self.test_results:
                    self.test_results[test_key] = result
                    reward_val += 0.15
                    reward_reasons.append(f"Ran {action.test_name} on {ds_id}: {result}")
                    if self._test_detects_fabrication(action.test_name, ds, self.paper.ground_truth_fabrication):
                        reward_val += 0.2
                        reward_reasons.append("Test correctly flagged anomaly!")
                else:
                    reward_val -= 0.1
                    reward_reasons.append("Test already run on this dataset")
            else:
                reward_val -= 0.2
                reward_reasons.append("Invalid test or no raw data available")
        
        elif action.action_type == "request_author_explanation":
            if self.available_raw_data:
                ds_id = self.available_raw_data[-1]
                explanation = self.paper.author_explanations.get(ds_id, "No response.")
                self.author_responses.append(f"Author on {ds_id}: {explanation}")
                reward_val += 0.05
                reward_reasons.append(f"Received author explanation: {explanation[:50]}...")
            else:
                reward_val -= 0.1
                reward_reasons.append("No dataset to ask about")
        
        elif action.action_type == "flag_paper":
            if action.severity and 1 <= action.severity <= 5:
                self.flags.append({"severity": action.severity, "step": self.step_count})
                reward_val += 0.1
                reward_reasons.append(f"Raised flag with severity {action.severity}")
            else:
                reward_val -= 0.2
                reward_reasons.append("Invalid severity")
        
        elif action.action_type == "issue_verdict":
            if action.verdict in ["retract", "require_revision", "accept"]:
                self.final_verdict = action.verdict
                self.done = True
                reward_reasons.append(f"Issued verdict: {action.verdict}")
            else:
                reward_val -= 0.2
                reward_reasons.append("Invalid verdict")
        
        if self.step_count >= 15:
            self.done = True
            if not self.final_verdict:
                self.final_verdict = "accept"
                reward_reasons.append("Timeout: default verdict 'accept'")
        
        if self.done:
            agent_log = {
                "datasets_requested": self.datasets_requested,
                "tests_run": list(self.test_results.keys()),
                "flags": self.flags,
                "final_verdict": self.final_verdict,
                "confidence": 0.8
            }
            final_score = Grader.grade(self.paper, agent_log)
            final_reward = final_score * 2.0
            total_reward = self.reward_total + reward_val + final_reward
            reward_val = total_reward
            reward_reasons.append(f"Final task score: {final_score} -> +{final_reward}")
            info["final_score"] = final_score
            info["verdict"] = self.final_verdict
        
        self.reward_total += reward_val
        reward_obj = Reward(value=reward_val, reasons=reward_reasons)
        obs = self._get_observation()
        return obs, reward_obj, self.done, info
    
    def _run_test(self, test_name: str, dataset: RawDataset) -> str:
        data = dataset.data
        if test_name == "benford":
            vals = data.get("values") or data.get("reaction_times", [])
            if not vals:
                return "Cannot apply Benford test to this data"
            first_digits = [int(str(abs(v)).strip('.')[0]) for v in vals if v > 0]
            expected = [np.log10(1+1/d) for d in range(1,10)]
            observed = [first_digits.count(d)/len(first_digits) for d in range(1,10)]
            chi2, p = stats.chisquare(observed, expected)
            if p < 0.05:
                return f"Benford test: significant deviation (p={p:.3f}) - possible fabrication"
            else:
                return f"Benford test: no significant deviation (p={p:.3f})"
        
        elif test_name == "digit_frequency":
            vals = data.get("values") or data.get("reaction_times", [])
            last_digits = [int(str(v).split('.')[-1][0]) for v in vals if str(v).find('.')>0]
            observed = [last_digits.count(d) for d in range(10)]
            expected = [len(last_digits)/10]*10
            chi2, p = stats.chisquare(observed, expected)
            if p < 0.05:
                return f"Digit frequency: suspicious uniformity (p={p:.3f})"
            else:
                return f"Digit frequency: ok (p={p:.3f})"
        
        elif test_name == "correlation_check":
            if "efficiency" in data and "temperature" in data:
                corr = np.corrcoef(data["efficiency"], data["temperature"])[0,1]
                if abs(corr) > 0.99:
                    return f"Impossible correlation detected: r={corr:.3f}"
                else:
                    return f"Correlation r={corr:.3f} (normal)"
            else:
                return "Correlation check not applicable"
        
        elif test_name == "timestamp_consistency":
            if "timestamps" in data:
                timestamps = data["timestamps"]
                if len(set(timestamps)) != len(timestamps):
                    return "Duplicate timestamps detected - possible data reuse"
                else:
                    return "Timestamps are unique"
            else:
                return "No timestamps available"
        
        elif test_name == "outlier_detection":
            vals = data.get("values") or data.get("efficiency", [])
            if vals:
                q75, q25 = np.percentile(vals, [75,25])
                iqr = q75 - q25
                outliers = sum(1 for v in vals if v < q25 - 1.5*iqr or v > q75 + 1.5*iqr)
                if outliers > len(vals)*0.1:
                    return f"High outlier count: {outliers}/{len(vals)}"
                else:
                    return f"Outlier count normal: {outliers}/{len(vals)}"
            else:
                return "Outlier detection not applicable"
        
        return "Test completed"
    
    def _test_detects_fabrication(self, test_name: str, dataset: RawDataset, gt_fabrication: Dict) -> bool:
        fab_type = dataset.fabrication_type
        if test_name == "outlier_detection" and fab_type == "duplicate_rows":
            return True
        if test_name in ["benford", "digit_frequency"] and fab_type == "benford_violation":
            return True
        if test_name == "correlation_check" and fab_type == "impossible_correlation":
            return True
        if test_name == "timestamp_consistency" and fab_type == "timestamp_reuse":
            return True
        return False
    
    def state(self) -> Dict[str, Any]:
        return {
            "paper": self.paper.title,
            "step_count": self.step_count,
            "datasets_requested": self.datasets_requested,
            "available_raw_data": self.available_raw_data,
            "test_results": self.test_results,
            "author_responses": self.author_responses,
            "flags": self.flags,
            "final_verdict": self.final_verdict,
            "done": self.done
        }