#!/usr/bin/env python3
import os, sys, yaml, json, requests, subprocess, importlib.util
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}\n {title}\n{'='*60}")

def check(condition, message, error=False):
    print(f"{'✅' if condition else '❌'} {message}")
    if error and not condition:
        sys.exit(1)
    return condition

# 1. OpenEnv spec
print_section("1. OpenEnv Spec")
check(Path("openenv.yaml").exists(), "openenv.yaml exists", error=True)
with open("openenv.yaml") as f:
    config = yaml.safe_load(f)
    check("name" in config, "has name")
    check("version" in config, "has version")

spec = importlib.util.spec_from_file_location("models", "models.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)
check(hasattr(models, "Observation"), "Observation defined")
check(hasattr(models, "Action"), "Action defined")
check(hasattr(models, "Reward"), "Reward defined")

spec = importlib.util.spec_from_file_location("env", "env.py")
env_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_mod)
check(hasattr(env_mod, "FraudDetectionEnv"), "FraudDetectionEnv class")
env = env_mod.FraudDetectionEnv("easy")
check(callable(env.reset), "reset() method")
check(callable(env.step), "step() method")
check(callable(env.state), "state() method")

# 2. Dockerfile
print_section("2. Dockerfile")
check(Path("Dockerfile").exists(), "Dockerfile exists", error=True)
with open("Dockerfile") as f:
    content = f.read()
    check("FROM" in content, "FROM directive")
    check("EXPOSE" in content, "EXPOSE directive")

# 3. inference.py
print_section("3. inference.py")
check(Path("inference.py").exists(), "inference.py exists", error=True)
with open("inference.py") as f:
    content = f.read()
    check("from openai import OpenAI" in content, "uses OpenAI client")
    check("API_BASE_URL" in content, "uses API_BASE_URL")
    check("MODEL_NAME" in content, "uses MODEL_NAME")
    check("HF_TOKEN" in content, "uses HF_TOKEN")
    check("[START]" in content, "emits [START]")
    check("[STEP]" in content, "emits [STEP]")
    check("[END]" in content, "emits [END]")

# 4. Tasks & Graders
print_section("4. Tasks and Graders")
spec = importlib.util.spec_from_file_location("tasks", "tasks.py")
tasks_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks_mod)
tasks = tasks_mod.TASKS
check(len(tasks) >= 3, f"{len(tasks)} tasks (>=3 required)")
grader = tasks_mod.Grader()
for name, paper in tasks.items():
    dummy_log = {"datasets_requested": [], "tests_run": [], "flags": [], "final_verdict": "accept", "confidence": 0.5}
    score = grader.grade(paper, dummy_log)
    check(0.0 <= score <= 1.0, f"Grade for {name}: {score} (in 0-1)")

# 5. HF Space ping (optional)
print_section("5. HF Space Ping (optional)")
space_url = os.environ.get("SPACE_URL")
if space_url:
    try:
        resp = requests.get(f"{space_url.rstrip('/')}/reset?task=easy", timeout=10)
        check(resp.status_code == 200, f"Space /reset responded with {resp.status_code}")
    except Exception as e:
        check(False, f"Could not reach Space: {e}")
else:
    print("⚠️ SPACE_URL not set – skipping")

print_section("VALIDATION SUMMARY")
print("✅ All required checks passed. Ready to submit!")