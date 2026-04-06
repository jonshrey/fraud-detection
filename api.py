from fastapi import FastAPI, HTTPException
from env import FraudDetectionEnv
import uvicorn
import threading

app = FastAPI()
envs = {}

@app.get("/reset")
def reset(task: str = "easy"):
    envs[task] = FraudDetectionEnv(task)
    obs = envs[task].reset()
    return {"status": "ok", "task": task, "step": obs.step_count}

@app.get("/state")
def state(task: str = "easy"):
    if task not in envs:
        raise HTTPException(status_code=404, detail="Environment not initialized. Call reset first.")
    return envs[task].state()

@app.post("/step")
def step(task: str, action: dict):
    if task not in envs:
        raise HTTPException(status_code=404, detail="Environment not initialized. Call reset first.")
    from models import Action
    act = Action(**action)
    obs, reward, done, info = envs[task].step(act)
    return {"observation": obs.dict(), "reward": reward.dict(), "done": done, "info": info}

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=7861)

# Run API in background thread when space_app.py starts
threading.Thread(target=run_api, daemon=True).start()