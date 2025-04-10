from sumo_env import SumoIntersectionEnv
from reinforce_agent import ReinforceAgent

# Load the environment with GUI
env = SumoIntersectionEnv("/Users/antoinechosson/Desktop/realistic_intersection/intersection.sumocfg", use_gui=False)

# Create agent and load trained model
agent = ReinforceAgent(env, gamma=0.99, lr=1e-4)
agent.load_model("reinforce_agent.pth")

# Run a single episode with visualization
env.use_gui = True
agent.env = env
agent.generate_episode(render=True)

env.close()

