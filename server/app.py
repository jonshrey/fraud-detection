from openenv_core import OpenEnvServer
from env import FraudDetectionEnv
from models import Observation, Action, Reward

def main():
    """Entry point for openenv serve (required name: main)."""
    env = FraudDetectionEnv(task_name="easy")
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
    main()