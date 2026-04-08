from typing import Dict, Any, List
from models import Observation, Action, Reward
from tasks import TASKS as TASK_REGISTRY
import numpy as np
from scipy import stats


class FraudDetectionEnv:
    def __init__(self, task_name: str = "easy"):
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Task must be one of {list(TASK_REGISTRY.keys())}")
        self.task_name = task_name
        self.paper = TASK_REGISTRY[task_name]["task"]
        self.reset()

    # ---------------- RESET ----------------
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

    # ---------------- OBSERVATION ----------------
    def _get_observation(self) -> Observation:
        available_desc = []

        for ds_id in self.available_raw_data:
            ds = next((d for d in self.paper.raw_datasets if d.id == ds_id), None)
            if ds:
                preview = str(ds.data)[:200]
                available_desc.append(f"{ds.id}: {ds.description} | preview: {preview}")

        return Observation(
            paper_metadata={
                "title": self.paper.title,
                "authors": self.paper.authors,
                "journal": self.paper.journal,
                "field": self.paper.field,
            },
            published_stats=self.paper.published_stats,
            available_raw_data=available_desc,
            test_results=self.test_results,
            author_responses=self.author_responses,
            step_count=self.step_count,
        )

    # ---------------- STEP ----------------
    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")

        self.step_count += 1

        # 🔥 Small penalty to discourage useless actions
        reward_val = -0.01
        reward_reasons = []
        info = {}

        # ---------- ACTION LOGIC ----------
        if action.action_type == "request_raw_data":
            if action.dataset_id:
                ds = next((d for d in self.paper.raw_datasets if d.id == action.dataset_id), None)

                if ds and action.dataset_id not in self.datasets_requested:
                    self.datasets_requested.append(action.dataset_id)
                    self.available_raw_data.append(action.dataset_id)
                    reward_val += 0.08
                    reward_reasons.append(f"Requested dataset {action.dataset_id}")

                elif ds:
                    reward_val -= 0.05
                    reward_reasons.append("Dataset already requested")

                else:
                    reward_val -= 0.1
                    reward_reasons.append("Invalid dataset ID")

            else:
                reward_val -= 0.1
                reward_reasons.append("Missing dataset_id")

        elif action.action_type == "run_statistical_test":
            if action.test_name and self.available_raw_data:
                ds_id = self.available_raw_data[-1]
                ds = next(d for d in self.paper.raw_datasets if d.id == ds_id)

                test_key = f"{ds_id}_{action.test_name}"

                if test_key not in self.test_results:
                    result = self._run_test(action.test_name, ds)
                    self.test_results[test_key] = result

                    reward_val += 0.12
                    reward_reasons.append(f"Ran {action.test_name}")

                    if self._test_detects_fabrication(action.test_name, ds):
                        reward_val += 0.15
                        reward_reasons.append("Correct anomaly detection")

                else:
                    reward_val -= 0.05
                    reward_reasons.append("Test already run")

            else:
                reward_val -= 0.1
                reward_reasons.append("Invalid test or no data")

        elif action.action_type == "request_author_explanation":
            if self.available_raw_data:
                ds_id = self.available_raw_data[-1]
                explanation = self.paper.author_explanations.get(ds_id, "No response.")
                self.author_responses.append(explanation)

                reward_val += 0.04
                reward_reasons.append("Got author explanation")
            else:
                reward_val -= 0.05

        elif action.action_type == "flag_paper":
            if action.severity and 1 <= action.severity <= 5:
                self.flags.append({"severity": action.severity, "step": self.step_count})
                reward_val += 0.05
                reward_reasons.append("Flag raised")
            else:
                reward_val -= 0.1

        elif action.action_type == "issue_verdict":
            if action.verdict in ["retract", "require_revision", "accept"]:
                self.final_verdict = action.verdict
                self.done = True
                reward_reasons.append(f"Verdict issued: {action.verdict}")
            else:
                reward_val -= 0.1

        # ---------- EPISODE END ----------
        if self.step_count >= 12:
            self.done = True
            if not self.final_verdict:
                self.final_verdict = "accept"

        # ---------- FINAL SCORING ----------
        if self.done:
            task_dict = TASK_REGISTRY[self.task_name]

            agent_log = {
                "datasets_requested": self.datasets_requested,
                "tests_run": list(self.test_results.keys()),
                "flags": self.flags,
                "final_verdict": self.final_verdict,
                "fabrication_type": self._infer_fabrication(),
                "location": self._infer_location(),
                "confidence": 0.8,
            }

            final_score = task_dict["grader"](task_dict["task"], agent_log)

            # 🔥 CRITICAL: normalize reward to [0,1]
            reward_val = float(final_score)

            info["final_score"] = final_score
            info["verdict"] = self.final_verdict

        self.reward_total += reward_val

        return (
            self._get_observation(),
            Reward(value=reward_val, reasons=reward_reasons),
            self.done,
            info,
        )

    # ---------------- TESTS ----------------
    def _run_test(self, test_name: str, dataset) -> str:
        data = dataset.data

        try:
            if test_name == "benford":
                vals = data.get("values") or data.get("reaction_times", [])
                if not vals:
                    return "No valid data"

                first_digits = [int(str(abs(v))[0]) for v in vals if v > 0]

                expected = [np.log10(1 + 1 / d) for d in range(1, 10)]
                observed = [first_digits.count(d) / len(first_digits) for d in range(1, 10)]

                _, p = stats.chisquare(observed, expected)
                return "benford_violation" if p < 0.05 else "benford_ok"

            elif test_name == "correlation_check":
                if "efficiency" in data and "temperature" in data:
                    corr = np.corrcoef(data["efficiency"], data["temperature"])[0, 1]
                    return "impossible_correlation" if abs(corr) > 0.99 else "normal"

            elif test_name == "timestamp_consistency":
                if "timestamps" in data:
                    ts = data["timestamps"]
                    return "timestamp_reuse" if len(set(ts)) != len(ts) else "ok"

            elif test_name == "outlier_detection":
                vals = data.get("values") or data.get("efficiency", [])
                if vals:
                    q75, q25 = np.percentile(vals, [75, 25])
                    iqr = q75 - q25
                    outliers = sum(1 for v in vals if v < q25 - 1.5 * iqr or v > q75 + 1.5 * iqr)
                    return "duplicate_rows" if outliers > len(vals) * 0.1 else "normal"

        except Exception:
            return "test_failed"

        return "test_completed"

    # ---------------- HELPERS ----------------
    def _test_detects_fabrication(self, test_name, dataset):
        return dataset.fabrication_type in ["duplicate_rows", "benford_violation", "impossible_correlation", "timestamp_reuse"]

    def _infer_fabrication(self):
        for result in self.test_results.values():
            if result in ["duplicate_rows", "benford_violation", "impossible_correlation", "timestamp_reuse"]:
                return result
        return ""

    def _infer_location(self):
        for key, result in self.test_results.items():
            if result in ["duplicate_rows", "benford_violation", "impossible_correlation", "timestamp_reuse"]:
                return key.split("_")[0]
        return ""

    # ---------------- STATE ----------------
    def state(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "datasets_requested": self.datasets_requested,
            "test_results": self.test_results,
            "flags": self.flags,
            "final_verdict": self.final_verdict,
            "done": self.done,
        }