import pickle
import random
import argparse
import mxnet as mx
from environment import run
from noises import OrnsteinUhlenbeckNoise
from utils import AgentBase
from pendulum_ddpg import Actor, Critic, Test


class Agent(AgentBase):
    def __init__(self, gamma=0.99, tau=5e-3, lambda1=1e-3, lambda2=1.0, random_steps=10000, batch_size=64, ctx=mx.cpu()):
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
        self.__demo = []
        self.__cache = []
        self.__noise = OrnsteinUhlenbeckNoise((1, 1), ctx=ctx)
        self.__gamma = gamma
        self.__tau = tau
        self.__lambda1 = lambda1
        self.__lambda2 = lambda2
        self.__random_steps = random_steps
        self.__batch_size = batch_size
        self.__context = ctx

    @property
    def test_agent(self):
        return Test(self.__actor, self.__context)

    def load_demo(self, demo):
        with open(demo, "rb") as f:
            data = pickle.load(f)
        for episode in data:
            for t in range(len(episode) - 1):
                state, _, action = episode[t]
                next_state, reward, _ = episode[t+1]
                s = mx.nd.array(state, ctx=self.__context).expand_dims(0)
                a = mx.nd.array(action, ctx=self.__context).expand_dims(0)
                s1 = mx.nd.zeros_like(s) if next_state is None else mx.nd.array(next_state, ctx=self.__context).expand_dims(0)
                self.__demo.append((s, a, s1, reward, float(not next_state is None)))

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
        ds, da, ds1, dr, dmask = self.__batch(self.__demo)
        with mx.autograd.record():
            pg_loss = -self.__critic(s, self.__actor(s))
            pda = self.__actor(ds)
            qf = self.__critic(ds, da) > self.__critic(ds, pda)
            bc_loss = mx.nd.mean(mx.nd.smooth_l1(mx.nd.abs(pda - da)), axis=0, keepdims=True, exclude=True) * qf
            L = self.__lambda1 * pg_loss + self.__lambda2 * bc_loss
            L.backward()
        self.__actor_trainer.step(self.__batch_size)
        self.__soft_update()

    def __batch(self, cache=None):
        if cache is None:
            cache = self.__cache
        s, a, s1, r, mask = zip(*random.sample(cache, k=self.__batch_size))
        return mx.nd.concat(*s, dim=0), mx.nd.concat(*a, dim=0), mx.nd.concat(*s1, dim=0), mx.nd.array(r, ctx=self.__context).expand_dims(1), mx.nd.array(mask, ctx=self.__context).expand_dims(1)

    def __soft_update(self):
        for action, target in [(self.__actor, self.__actor_target), (self.__critic, self.__critic_target)]:
            for name, param in target.collect_params().items():
                param.set_data((1 - self.__tau) * param.data(self.__context) + self.__tau * action.collect_params().get(name.removeprefix(target.prefix)).data(self.__context))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of BC for Pendulum-v1.")
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
