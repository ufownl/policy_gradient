import random
import argparse
import mxnet as mx
from environment import run
from utils import AgentBase
from pendulum_a2c import Actor, Critic, Test


class Agent(AgentBase):
    def __init__(self, gamma=0.9, epsilon=0.2, tau=0.8, entropy_weight=5e-3, rollout_length=2000, epochs=64, batch_size=64, ctx=mx.cpu()):
        super(Agent, self).__init__("Pendulum-v1")
        self.__actor = Actor()
        self.__actor.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__actor_trainer = mx.gluon.Trainer(self.__actor.collect_params(), "Nadam", {
            "learning_rate": 1e-4
        })
        self.__critic = Critic()
        self.__critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic_trainer = mx.gluon.Trainer(self.__critic.collect_params(), "Nadam", {
            "learning_rate": 1e-3
        })
        self.__cache = []
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__tau = tau
        self.__entropy_weight = entropy_weight
        self.__rollout_length = rollout_length
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__context = ctx

    @property
    def test_agent(self):
        return Test(self.__actor, self.__context)

    def __call__(self):
        state, _ = yield
        while not state is None:
            x = mx.nd.array(state, ctx=self.__context).expand_dims(0)
            d = self.__actor(x)
            y = d.sample()
            v = self.__critic(x)[0].asscalar()
            action = y.clip(-2.0, 2.0).asnumpy()[0]
            s1, r = yield action
            self.__cache.append((x, v, y, mx.nd.log(d.probability(y)), s1, r))
            if len(self.__cache) >= self.__rollout_length:
                self.__estimate_g(s1)
                self.__update_model()
                self.__cache = []
            state = s1

    def __estimate_g(self, last_state):
        values = [v for _, v, _, _, _, _ in self.__cache] + [0 if last_state is None else self.__critic(mx.nd.array(last_state, ctx=self.__context).expand_dims(0))[0].asscalar()]
        gae = 0
        for i in reversed(range(len(self.__cache))):
            _, _, _, _, s1, r = self.__cache[i]
            delta = r - values[i] if s1 is None else r + self.__gamma * values[i + 1] - values[i]
            gae = delta if s1 is None else delta + self.__gamma * self.__tau * gae
            self.__cache[i] += (values[i] + gae,)

    def __update_model(self):
        for x, v, y, p, g in self.__batches():
            advantage = g - v
            with mx.autograd.record():
                d = self.__actor(x)
                ratio = (mx.nd.log(d.probability(y)) - p).exp()
                L = -mx.nd.min(mx.nd.concat(ratio * advantage, ratio.clip(1.0 - self.__epsilon, 1.0 + self.__epsilon) * advantage, dim=1), axis=1, keepdims=True) - self.__entropy_weight * d.entropy
                L.backward()
            self.__actor_trainer.step(self.__batch_size)
            with mx.autograd.record():
                L = mx.nd.smooth_l1(mx.nd.abs(g - self.__critic(x)))
                L.backward()
            self.__critic_trainer.step(self.__batch_size)

    def __batches(self):
        for _ in range(self.__epochs):
            for _ in range(len(self.__cache) // self.__batch_size):
                x, v, y, p, _, _, g = zip(*random.sample(self.__cache, k=self.__batch_size))
                yield mx.nd.concat(*x, dim=0), mx.nd.array(v, ctx=self.__context).expand_dims(1), mx.nd.concat(*y, dim=0), mx.nd.concat(*p, dim=0), mx.nd.array(g, ctx=self.__context).expand_dims(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of PPO for Pendulum-v1.")
    parser.add_argument("--episodes", help="number of training episodes (default: 500)", type=int, default=500)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        agent = Agent(ctx=mx.gpu(args.device_id))
    else:
        agent = Agent(ctx=mx.cpu(args.device_id))
    print("Training...", flush=True)
    run(agent, args.episodes)
    print("Testing...", flush=True)
    run(agent.test_agent, 5)
    print("Done!", flush=True)
