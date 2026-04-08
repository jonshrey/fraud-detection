import os
import json
from openai import OpenAI
from env import FraudDetectionEnv
from tasks import PAPERS, Grader
from models import Action

def run_agent_on_task(task_name: str) -> float:
    # Initialize OpenAI client with hackathon's LiteLLM proxy
    # NO fallback - must use hackathon credentials
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
    
    env = FraudDetectionEnv(task_name)
    obs = env.reset()
    done = False
    step_num = 0
    
    conversation_history = []
    
    # System prompt to guide the LLM
    system_prompt = """You are an AI agent tasked with detecting fraud in scientific papers.
Available actions:
- request_raw_data: {"action_type": "request_raw_data", "dataset_id": "<id>"}
- run_statistical_test: {"action_type": "run_statistical_test", "test_name": "<test>"}
  Tests: benford, digit_frequency, correlation_check, timestamp_consistency, outlier_detection
- request_author_explanation: {"action_type": "request_author_explanation"}
- flag_paper: {"action_type": "flag_paper", "severity": 1-5}
- issue_verdict: {"action_type": "issue_verdict", "verdict": "retract|require_revision|accept"}

Respond with ONLY a valid JSON action object."""
    
    while not done and step_num < 15:
        step_num += 1
        
        # Prepare observation for LLM
        obs_text = f"""Paper: {obs.paper_metadata['title']}
Authors: {obs.paper_metadata['authors']}
Published Stats: {json.dumps(obs.published_stats, indent=2)}
Available Datasets: {json.dumps([d for d in env.paper.raw_datasets], default=lambda x: x.id if hasattr(x, 'id') else str(x))}
Already Requested: {env.datasets_requested}
Test Results: {json.dumps(obs.test_results, indent=2)}
Step: {obs.step_count}/15"""
        
        # Call LLM through the proxy
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or whatever model the hackathon supports
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_text}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content.strip()
            print(f"[LLM] {llm_response}")
            
            # Parse the JSON action
            # Remove markdown code blocks if present
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1].split("```")[0].strip()
            
            action_dict = json.loads(llm_response)
            action = Action(**action_dict)
            
        except Exception as e:
            print(f"[ERROR] LLM parse failed: {e}")
            # Fallback action
            if step_num == 1:
                action = Action(action_type="request_raw_data", dataset_id=env.paper.raw_datasets[0].id)
            else:
                action = Action(action_type="issue_verdict", verdict="accept")
        
        obs, reward, done, info = env.step(action)
        print(f"[STEP] task={task_name} step={step_num} action={action.action_type} reward={reward.value:.2f}")
    
    final_score = info.get("final_score", 0.5)
    return final_score

def main():
    print("[START] fraud-detection-env")
    scores = {}
    for task in ["easy", "medium", "hard"]:
        try:
            scores[task] = run_agent_on_task(task)
        except Exception as e:
            print(f"Error on {task}: {e}")
            scores[task] = 0.5
    avg = sum(scores.values()) / len(scores)
    print(f"[END] easy={scores['easy']:.2f} medium={scores['medium']:.2f} hard={scores['hard']:.2f} avg={avg:.2f}")

if __name__ == "__main__":
    main()