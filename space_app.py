import gradio as gr
import os
import json
from fastapi import FastAPI
import uvicorn
from env import FraudDetectionEnv
from models import Action

# ---------- Gradio UI function ----------
def run_agent_step(task_name, action_json):
    try:
        env = FraudDetectionEnv(task_name)
        obs = env.reset()
        action_data = json.loads(action_json)
        action = Action(**action_data)
        obs, reward, done, info = env.step(action)
        
        output = f"Step {obs.step_count}\n"
        output += f"Action: {action.action_type}\n"
        output += f"Reward: {reward.value:.2f}\n"
        output += f"Reasons: {', '.join(reward.reasons)}\n\n"
        output += f"Available raw data: {obs.available_raw_data}\n"
        output += f"Test results: {obs.test_results}\n"
        output += f"Author responses: {obs.author_responses}\n"
        if done:
            output += f"\nEpisode done! Final score: {info.get('final_score', 0)}\n"
            output += f"Verdict: {env.final_verdict}\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Research Integrity Officer - Fraud Detection") as demo:
    gr.Markdown("# 🔬 AI Research Integrity Officer\nInvestigate scientific papers for potential data fabrication.")
    with gr.Row():
        task_dropdown = gr.Dropdown(["easy", "medium", "hard"], label="Select Task Difficulty")
        action_input = gr.Textbox(label="Action JSON", lines=3, placeholder='{"action_type": "request_raw_data", "dataset_id": "raw_data_1"}')
    submit_btn = gr.Button("Execute Action")
    output_text = gr.Textbox(label="Environment Response", lines=15)
    submit_btn.click(run_agent_step, inputs=[task_dropdown, action_input], outputs=output_text)
    gr.Markdown("""
    ### Available Actions
    - `request_raw_data`: `{"action_type": "request_raw_data", "dataset_id": "raw_data_1"}`
    - `run_statistical_test`: `{"action_type": "run_statistical_test", "test_name": "benford"}`
    - `request_author_explanation`: `{"action_type": "request_author_explanation"}`
    - `flag_paper`: `{"action_type": "flag_paper", "severity": 3}`
    - `issue_verdict`: `{"action_type": "issue_verdict", "verdict": "retract"}`
    """)

# ---------- Create FastAPI app and mount Gradio ----------
app = FastAPI()

@app.api_route("/reset", methods=["GET", "POST"])
def reset(task: str = "easy"):
    env = FraudDetectionEnv(task)
    obs = env.reset()
    return {"status": "ok", "task": task, "step": obs.step_count}

# Mount the Gradio app at the root path "/"
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
