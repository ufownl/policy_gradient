import pickle
import random
import argparse
import mxnet as mx
from environment import run
from noises import OrnsteinUhlenbeckNoise
from segment_tree import SumSegmentTree, MinSegmentTree
from utils import AgentBase
from pendulum_ddpg import Actor, Critic, Test


class PrioritizedCache:
    def __init__(self, size, alpha, beta):
        self.__sum_st = SumSegmentTree(size)
        self.__min_st = MinSegmentTree(size)
        self.__buffer = [None] * size
        self.__alpha = alpha
        self.__beta = beta
        self.__size = 0
        self.__cursor = 0
        self.__max_priority = 1.0

    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, value):
        self.__beta = max(min(value, 1.0), 0.0)

    def append(self, experience):
        self.__sum_st[self.__cursor] = self.__max_priority
        self.__min_st[self.__cursor] = self.__max_priority
        self.__buffer[self.__cursor] = experience
        self.__size = min(self.__size + 1, len(self.__buffer))
        self.__cursor = (self.__cursor + 1) % len(self.__buffer)

    def sample(self, k):
        return [self.__sample_impl() for _ in range(k)]

    def update_priority(self, key, priority, epsilon=1e-6):
        p = (priority + epsilon) ** self.__alpha
        self.__sum_st[key] = p
        self.__min_st[key] = p
        self.__max_priority = max(p, self.__max_priority)

    def __len__(self):
        return self.__size

    def __sample_impl(self):
        key = self.__sum_st.find_prefixsum_idx(random.uniform(0.0, self.__sum_st.sum()))
        p = self.__sum_st[key] / self.__sum_st.sum()
        w = (p * self.__size) ** -self.__beta
        min_p = self.__min_st.min() / self.__sum_st.sum()
        max_w = (min_p * self.__size) ** -self.__beta
        return key, w / max_w, self.__buffer[key]


class Agent(AgentBase):
    def __init__(self, gamma=0.99, tau=5e-3, random_steps=10000, n_step=3, batch_size=64, ctx=mx.cpu()):
        super(Agent, self).__init__("Pendulum-v1")
        self.__actor = Actor()
        self.__actor.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__actor_trainer = mx.gluon.Trainer(self.__actor.collect_params(), "Nadam", {
            "learning_rate": 1e-4,
            "wd": 1e-4
        })
        self.__actor_target = Actor()
        self.__actor_target.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic = Critic()
        self.__critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic_trainer = mx.gluon.Trainer(self.__critic.collect_params(), "Nadam", {
            "learning_rate": 1e-3,
            "wd": 1e-4
        })
        self.__critic_target = Critic()
        self.__critic_target.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__cache = PrioritizedCache(1024*1024, 0.3, 1.0)
        self.__noise = OrnsteinUhlenbeckNoise((1, 1), ctx=ctx)
        self.__gamma = gamma
        self.__tau = tau
        self.__random_steps = random_steps
        self.__n_step = n_step
        self.__batch_size = batch_size
        self.__context = ctx

    @property
    def test_agent(self):
        return Test(self.__actor, self.__context)

    def load_demo(self, demo, epsilon=1.0):
        with open(demo, "rb") as f:
            data = pickle.load(f)
        for episode in data:
            t = self.__n_step - 1
            while True:
                update_t = t + 1 - self.__n_step
                state, _, action = episode[update_t]
                next_state, _, _ = episode[t+1] if t + 1 < len(episode) else episode[-1]
                reward = sum(r for _, r, _ in episode[update_t+1:t+2])
                s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
                a = mx.nd.array(action, ctx=self.__context).expand_dims(0)
                s1 = mx.nd.zeros_like(s) if next_state is None else mx.nd.array(next_state, ctx=self.__context).expand_dims(0)
                self.__cache.append((s, a, s1, reward, float(not next_state is None), epsilon))
                if episode[update_t + 1][0] is None:
                    break
                t += 1

    def __call__(self):
        state, _ = yield
        if state is None:
            return
        if len(self.__cache) < self.__random_steps:
            action = self.spaces[1].sample()
        else:
            s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
            a = (self.__actor(s) + self.__noise.sample()).clip(-2.0, 2.0)
            action = a.asnumpy()[0]
        episode = [(state, None, action)]
        t = 0
        while True:
            state, _, action = episode[-1]
            if not state is None:
                next_state, reward = yield action
                if next_state is None:
                    episode.append((None, reward, None))
                elif len(self.__cache) < self.__random_steps:
                    episode.append((next_state, reward, self.spaces[1].sample()))
                else:
                    s1 = mx.nd.array(next_state, ctx=self.__context).expand_dims(0)
                    a1 = (self.__actor(s1) + self.__noise.sample()).clip(-2.0, 2.0)
                    episode.append((next_state, reward, a1.asnumpy()[0]))
            update_t = t + 1 - self.__n_step
            if update_t >= 0:
                state, _, action = episode[update_t]
                next_state, _, _ = episode[-1]
                reward = sum(r for _, r, _ in episode[update_t+1:])
                s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
                a = mx.nd.array(action, ctx=self.__context).expand_dims(0)
                s1 = mx.nd.zeros_like(s) if next_state is None else mx.nd.array(next_state, ctx=self.__context).expand_dims(0)
                self.__cache.append((s, a, s1, reward, float(not next_state is None), 0.0))
                if len(self.__cache) >= self.__batch_size:
                    self.__update_model()
                if episode[update_t + 1][0] is None:
                    break
            t += 1
        self.__noise.reset()

    def __update_model(self):
        k, w, s, a, s1, r, mask, eps = self.__batch()
        g = r + self.__gamma * self.__critic_target(s1, self.__actor_target(s1)) * mask
        with mx.autograd.record():
            critic_loss = mx.nd.smooth_l1(mx.nd.abs(g - self.__critic(s, a)))
            L = critic_loss * w
            L.backward()
        self.__critic_trainer.step(self.__batch_size)
        with mx.autograd.record():
            actor_loss = -self.__critic(s, self.__actor(s))
            L = actor_loss * w
            L.backward()
        self.__actor_trainer.step(self.__batch_size)
        self.__soft_update()
        p = (critic_loss + actor_loss ** 2 + eps).reshape((-1,)).asnumpy()
        for key, priority in zip(k, p):
            self.__cache.update_priority(key, priority)

    def __batch(self):
        k, w, batch = zip(*self.__cache.sample(self.__batch_size))
        s, a, s1, r, mask, eps = zip(*batch)
        return k, mx.nd.array(w, ctx=self.__context).expand_dims(1), mx.nd.concat(*s, dim=0), mx.nd.concat(*a, dim=0), mx.nd.concat(*s1, dim=0), mx.nd.array(r, ctx=self.__context).expand_dims(1), mx.nd.array(mask, ctx=self.__context).expand_dims(1), mx.nd.array(eps, ctx=self.__context).expand_dims(1)

    def __soft_update(self):
        for action, target in [(self.__actor, self.__actor_target), (self.__critic, self.__critic_target)]:
            for name, param in target.collect_params().items():
                param.set_data((1 - self.__tau) * param.data(self.__context) + self.__tau * action.collect_params().get(name.removeprefix(target.prefix)).data(self.__context))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of DDPGfD for Pendulum-v1.")
    parser.add_argument("--episodes", help="number of training episodes (default: 500)", type=int, default=500)
    parser.add_argument("--demo", help="file path of demonstrations (default: demo.pkl)", type=str, default="demo.pkl")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        agent = Agent(ctx=mx.gpu(args.device_id))
    else:
        agent = Agent(ctx=mx.cpu(args.device_id))
    agent.load_demo(args.demo)
    print("Training...", flush=True)
    run(agent, args.episodes)
    print("Testing...", flush=True)
    run(agent.test_agent, 5)
    print("Done!", flush=True)
