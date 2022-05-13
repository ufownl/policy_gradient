import gym
from tqdm import tqdm


def run(agent, episodes, before_episode=None, after_episode=None, **kwargs):
    env = gym.make(agent.environment)
    agent.spaces = (env.observation_space, env.action_space)
    for i in tqdm(range(episodes)):
        if not before_episode is None:
            before_episode(i + 1)
        g = agent(**kwargs)
        next(g)
        state = env.reset()
        reward = 0
        while True:
            if agent.test:
                env.render()
            action = g.send((state, reward))
            state, reward, done, _ = env.step(action)
            if done:
                try:
                    g.send((None, reward))
                except StopIteration:
                    pass
                break
        if not after_episode is None:
            after_episode(i + 1)
    env.close()
