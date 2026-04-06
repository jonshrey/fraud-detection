import os
import json
from openai import OpenAI
from env import FraudDetectionEnv
from tasks import TASKS
from models import Action

# Required environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_agent_on_task(task_name: str) -> float:
    env = FraudDetectionEnv(task_name)
    obs = env.reset()
    done = False
    step_num = 0
    
    while not done:
        step_num += 1
        prompt = f"""You are an AI research integrity officer investigating potential scientific fraud.

PAPER: {obs.paper_metadata['title']} by {obs.paper_metadata['authors']} in {obs.paper_metadata['journal']}
FIELD: {obs.paper_metadata['field']}
PUBLISHED STATS: {obs.published_stats}

RAW DATA AVAILABLE: {obs.available_raw_data if obs.available_raw_data else 'None requested yet'}

TEST RESULTS: {obs.test_results}

AUTHOR RESPONSES: {obs.author_responses}

STEP: {obs.step_count}

Actions (choose one):
- request_raw_data with dataset_id (options: {[d.id for d in env.paper.raw_datasets]})
- run_statistical_test with test_name (benford, digit_frequency, outlier_detection, correlation_check, timestamp_consistency)
- request_author_explanation (no params)
- flag_paper with severity (1-5)
- issue_verdict with verdict (retract, require_revision, accept)

Respond with a JSON action. Example: {{"action_type": "request_raw_data", "dataset_id": "raw_data_1"}}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            action_data = json.loads(response.choices[0].message.content)
            action = Action(**action_data)
        except Exception as e:
            print(f"Invalid action: {e}, defaulting to request_author_explanation")
            action = Action(action_type="request_author_explanation")
        
        obs, reward, done, info = env.step(action)
        print(f"[STEP] task={task_name} step={step_num} action={action.action_type} reward={reward.value:.2f}")
    
    final_score = info.get("final_score", 0.0)
    return final_score

def main():
    print("[START] fraud-detection-env")
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_agent_on_task(task)
    avg = sum(scores.values()) / len(scores)
    print(f"[END] easy={scores['easy']:.2f} medium={scores['medium']:.2f} hard={scores['hard']:.2f} avg={avg:.2f}")

if __name__ == "__main__":
    main()