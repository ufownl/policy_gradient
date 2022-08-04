import pickle
import random
import argparse
import mxnet as mx
from environment import run
from noises import OrnsteinUhlenbeckNoise
from utils import AgentBase


class Actor(mx.gluon.nn.Block):
    def __init__(self, dims=128, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.__net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self.__net.add(
                mx.gluon.nn.Dense(dims, activation="relu"),
                mx.gluon.nn.Dense(1, activation="tanh", weight_initializer=mx.initializer.Uniform())
            )

    def forward(self, x):
        return self.__net(x) * 2


class Critic(mx.gluon.nn.Block):
    def __init__(self, dims=128, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.__net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self.__net.add(
                mx.gluon.nn.Dense(dims, activation="relu"),
                mx.gluon.nn.Dense(1, weight_initializer=mx.initializer.Uniform())
            )

    def forward(self, s, a):
        x = mx.nd.concat(s, a, dim=1)
        return self.__net(x)


class Agent(AgentBase):
    def __init__(self, gamma=0.99, tau=5e-3, random_steps=10000, batch_size=64, ctx=mx.cpu()):
        super(Agent, self).__init__("Pendulum-v1")
        self.__actor = Actor()
        self.__actor.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__actor_trainer = mx.gluon.Trainer(self.__actor.collect_params(), "Nadam", {
            "learning_rate": 1e-4
        })
        self.__actor_target = Actor()
        self.__actor_target.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic = Critic()
        self.__critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic_trainer = mx.gluon.Trainer(self.__critic.collect_params(), "Nadam", {
            "learning_rate": 1e-3
        })
        self.__critic_target = Critic()
        self.__critic_target.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__cache = []
        self.__noise = OrnsteinUhlenbeckNoise((1, 1), ctx=ctx)
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
                a = (self.__actor(s) + self.__noise.sample()).clip(-2.0, 2.0)
                action = a.asnumpy()[0]
            s1, r = yield action
            self.__cache.append((s, a, mx.nd.zeros_like(s) if s1 is None else mx.nd.array(s1, ctx=self.__context).expand_dims(0), r, float(not s1 is None)))
            if len(self.__cache) >= self.__batch_size:
                self.__update_model()
            state = s1
        self.__noise.reset()

    def __update_model(self):
        s, a, s1, r, mask = self.__batch()
        g = r + self.__gamma * self.__critic_target(s1, self.__actor_target(s1)) * mask
        with mx.autograd.record():
            L = mx.nd.smooth_l1(mx.nd.abs(g - self.__critic(s, a)))
            L.backward()
        self.__critic_trainer.step(self.__batch_size)
        with mx.autograd.record():
            L = -self.__critic(s, self.__actor(s))
            L.backward()
        self.__actor_trainer.step(self.__batch_size)
        self.__soft_update()

    def __batch(self):
        s, a, s1, r, mask = zip(*random.sample(self.__cache, k=self.__batch_size))
        return mx.nd.concat(*s, dim=0), mx.nd.concat(*a, dim=0), mx.nd.concat(*s1, dim=0), mx.nd.array(r, ctx=self.__context).expand_dims(1), mx.nd.array(mask, ctx=self.__context).expand_dims(1)

    def __soft_update(self):
        for action, target in [(self.__actor, self.__actor_target), (self.__critic, self.__critic_target)]:
            for name, param in target.collect_params().items():
                param.set_data((1 - self.__tau) * param.data(self.__context) + self.__tau * action.collect_params().get(name.removeprefix(target.prefix)).data(self.__context))


class Test(AgentBase):
    def __init__(self, actor, ctx):
        super(Test, self).__init__("Pendulum-v1", True)
        self.__actor = actor
        self.__context = ctx
        self.__demo = []

    @property
    def demo(self):
        return self.__demo

    def __call__(self):
        episode = []
        state, reward = yield
        while not state is None:
            s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
            action = self.__actor(s).asnumpy()[0]
            episode.append((state, reward, action))
            s1, reward = yield action
            state = s1
        episode.append((None, reward, None))
        self.__demo.append(episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of DDPG for Pendulum-v1.")
    parser.add_argument("--episodes", help="number of training episodes (default: 500)", type=int, default=500)
    parser.add_argument("--demo", help="file path of demonstrations (default: demo.pkl)", type=str, default="demo.pkl")
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
    test = agent.test_agent
    run(test, 5)
    print("Dumping...", flush=True)
    with open(args.demo, "wb") as f:
        pickle.dump(test.demo, f)
    print("Done!", flush=True)
