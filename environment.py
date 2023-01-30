import gym
from tqdm import tqdm


def run(agent, episodes, before_episode=None, after_episode=None, **kwargs):
    env = gym.make(agent.environment, render_mode="human") if agent.test else gym.make(agent.environment)
    agent.spaces = (env.observation_space, env.action_space)
    for i in tqdm(range(episodes)):
        if not before_episode is None:
            before_episode(i + 1)
        g = agent(**kwargs)
        next(g)
        state, _ = env.reset()
        reward = 0
        while True:
            action = g.send((state, reward))
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                try:
                    g.send((None, reward))
                except StopIteration:
                    pass
                break
        if not after_episode is None:
            after_episode(i + 1)
    env.close()
