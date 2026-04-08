from openenv_core import OpenEnvServer
from env import FraudDetectionEnv
from models import Observation, Action, Reward

env = FraudDetectionEnv(task_name="easy")

server = OpenEnvServer(
    env=env,
    observation_model=Observation,
    action_model=Action,
    reward_model=Reward,
    host="0.0.0.0",
    port=7860,
)

if __name__ == "__main__":
    server.run()