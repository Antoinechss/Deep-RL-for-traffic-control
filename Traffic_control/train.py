from sumo_env import SumoIntersectionEnv
from reinforce_agent import ReinforceAgent

env = SumoIntersectionEnv("/Users/antoinechosson/Desktop/intersection/1tls_2x2.sumocfg",use_gui=False)
agent = ReinforceAgent(env, gamma=0.99, lr=1e-4)
agent.train(num_episodes=500)
