import argparse
import mxnet as mx
from environment import run
from distributions import Normal
from utils import AgentBase


class Actor(mx.gluon.nn.Block):
    def __init__(self, dims=128, **kwargs):
        super(Actor, self).__init__(**kwargs)
        with self.name_scope():
            self.__hidden = mx.gluon.nn.Dense(dims, activation="relu")
            self.__mu = mx.gluon.nn.Dense(1, activation="tanh", weight_initializer=mx.initializer.Uniform())
            self.__std = mx.gluon.nn.Dense(1, activation="softrelu", weight_initializer=mx.initializer.Uniform())

    def forward(self, x):
        y = self.__hidden(x)
        return Normal(self.__mu(y) * 2, mx.nd.exp(self.__std(y)))


class Critic(mx.gluon.nn.Block):
    def __init__(self, dims=128, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.__net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self.__net.add(
                mx.gluon.nn.Dense(dims, activation="relu"),
                mx.gluon.nn.Dense(1, weight_initializer=mx.initializer.Uniform())
            )

    def forward(self, x):
        return self.__net(x)


class Agent(AgentBase):
    def __init__(self, gamma=0.9, entropy_weight=5e-3, ctx=mx.cpu()):
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
        self.__gamma = gamma
        self.__entropy_weight = entropy_weight
        self.__context = ctx

    @property
    def test_agent(self):
        return Test(self.__actor, self.__context)

    def __call__(self):
        state, _ = yield
        while not state is None:
            s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
            a = self.__actor(s).sample()
            action = a.clip(-2.0, 2.0).asnumpy()[0]
            s1, r = yield action
            g = r if s1 is None else r + self.__gamma * self.__critic(mx.nd.array(s1, ctx=self.__context).expand_dims(0))
            advantage = g - self.__critic(s)
            with mx.autograd.record():
                d = self.__actor(s)
                L = -advantage * mx.nd.log(d.probability(a)) - self.__entropy_weight * d.entropy
                L.backward()
            self.__actor_trainer.step(1)
            with mx.autograd.record():
                L = mx.nd.smooth_l1(mx.nd.abs(g - self.__critic(s)))
                L.backward()
            self.__critic_trainer.step(1)
            state = s1


class Test(AgentBase):
    def __init__(self, actor, ctx):
        super(Test, self).__init__("Pendulum-v1", True)
        self.__actor = actor
        self.__context = ctx

    def __call__(self):
        state, _ = yield
        while not state is None:
            s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
            action = self.__actor(s).mean.asnumpy()[0]
            s1, _ = yield action
            state = s1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of A2C for Pendulum-v1.")
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
