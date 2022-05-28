import math
import random
import argparse
import mxnet as mx
from environment import run
from distributions import Normal
from utils import AgentBase
from pendulum_a2c import Actor, Test
from pendulum_ddpg import Critic


class Agent(AgentBase):
    def __init__(self, gamma=0.99, tau=5e-3, random_steps=10000, batch_size=64, ctx=mx.cpu()):
        super(Agent, self).__init__("Pendulum-v1")
        self.__actor = Actor()
        self.__actor.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__actor_trainer = mx.gluon.Trainer(self.__actor.collect_params(), "Nadam", {
            "learning_rate": 3e-4
        })
        self.__critics = [Critic() for _ in range(2)]
        for critic in self.__critics:
            critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic_trainers = [mx.gluon.Trainer(critic.collect_params(), "Nadam", {
            "learning_rate": 3e-4
        }) for critic in self.__critics]
        self.__critic_targets = [Critic() for _ in range(2)]
        for critic in self.__critic_targets:
            critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__alpha =  mx.gluon.Parameter("alpha", shape=(1,))
        self.__alpha.initialize(mx.initializer.Zero(), ctx=ctx)
        self.__alpha_trainer = mx.gluon.Trainer([self.__alpha], "Nadam", {
            "learning_rate": 3e-4
        })
        self.__cache = []
        self.__gamma = gamma
        self.__tau = tau
        self.__random_steps = random_steps
        self.__batch_size = batch_size
        self.__context = ctx

    @property
    def test_agent(self):
        return Test(self.__actor, self.__context)

    def __call__(self):
        state, _ = yield
        while not state is None:
            s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
            if len(self.__cache) < self.__random_steps:
                action = self.spaces[1].sample()
                a = mx.nd.array(action, ctx=self.__context).expand_dims(0)
            else:
                a = self.__actor(s).sample()
                action = a.clip(-2.0, 2.0).asnumpy()[0]
            s1, r = yield action
            self.__cache.append((s, a, mx.nd.zeros_like(s) if s1 is None else mx.nd.array(s1, ctx=self.__context).expand_dims(0), r, float(not s1 is None)))
            if len(self.__cache) >= self.__batch_size:
                self.__update_model()
            state = s1

    def __update_model(self):
        s, a, s1, r, mask = self.__batch()
        alpha = mx.nd.exp(self.__alpha.data(self.__context))
        d1 = self.__actor(s1)
        a1 = d1.sample()
        g = r + self.__gamma * (mx.nd.min(mx.nd.concat(*[critic(s1, a1) for critic in self.__critic_targets], dim=1), axis=1, keepdims=True) - alpha * d1.log_prob(a1)) * mask
        with mx.autograd.record():
            for critic in self.__critics:
                L = mx.nd.smooth_l1(mx.nd.abs(g - critic(s, a)))
                L.backward()
        for trainer in self.__critic_trainers:
            trainer.step(self.__batch_size)
        with mx.autograd.record():
            d0 = self.__actor(s)
            a0 = d0.rsample()
            p0 = d0.log_prob(a0)
            L = alpha * p0 - mx.nd.min(mx.nd.concat(*[critic(s, a0) for critic in self.__critics], dim=1), axis=1, keepdims=True)
            L.backward()
        self.__actor_trainer.step(self.__batch_size)
        with mx.autograd.record():
            L = -mx.nd.exp(self.__alpha.data(self.__context)) * (p0.detach() - mx.nd.prod(mx.nd.array(self.spaces[1].shape, ctx=self.__context)))
            L.backward()
        self.__alpha_trainer.step(self.__batch_size)
        self.__soft_update()

    def __batch(self):
        s, a, s1, r, mask = zip(*random.sample(self.__cache, k=self.__batch_size))
        return mx.nd.concat(*s, dim=0), mx.nd.concat(*a, dim=0), mx.nd.concat(*s1, dim=0), mx.nd.array(r, ctx=self.__context).expand_dims(1), mx.nd.array(mask, ctx=self.__context).expand_dims(1)

    def __soft_update(self):
        for action, target in zip(self.__critics, self.__critic_targets):
            for name, param in target.collect_params().items():
                param.set_data((1 - self.__tau) * param.data(self.__context) + self.__tau * action.collect_params().get(name.removeprefix(target.prefix)).data(self.__context))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of SAC for Pendulum-v1.")
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
