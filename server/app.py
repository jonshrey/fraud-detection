from openenv_core import OpenEnvServer
from env import FraudDetectionEnv
from models import Observation, Action, Reward

def run_server():
    """Entry point for openenv serve."""
    env = FraudDetectionEnv(task_name="easy")  # default task
    server = OpenEnvServer(
        env=env,
        observation_model=Observation,
        action_model=Action,
        reward_model=Reward,
        host="0.0.0.0",
        port=8000,
    )
    server.run()

if __name__ == "__main__":
    run_server()