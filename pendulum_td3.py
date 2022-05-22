import random
import argparse
import mxnet as mx
from environment import run
from noises import GaussNoise
from utils import AgentBase
from pendulum_ddpg import Actor, Critic, Test


class Agent(AgentBase):
    def __init__(self, gamma=0.99, tau=5e-3, target_noise_clip=(-0.5, 0.5), random_steps=10000, delayed_steps=2, batch_size=64, ctx=mx.cpu()):
        super(Agent, self).__init__("Pendulum-v1")
        self.__actor = Actor()
        self.__actor.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__actor_trainer = mx.gluon.Trainer(self.__actor.collect_params(), "Nadam", {
            "learning_rate": 1e-4
        })
        self.__actor_target = Actor()
        self.__actor_target.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critics = [Critic() for _ in range(2)]
        for critic in self.__critics:
            critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__critic_trainers = [mx.gluon.Trainer(critic.collect_params(), "Nadam", {
            "learning_rate": 1e-3
        }) for critic in self.__critics]
        self.__critic_targets = [Critic() for _ in range(2)]
        for critic in self.__critic_targets:
            critic.initialize(mx.initializer.Xavier(), ctx=ctx)
        self.__cache = []
        self.__noise = GaussNoise((1, 1), ctx=ctx)
        self.__gamma = gamma
        self.__tau = tau
        self.__target_noise_clip = target_noise_clip
        self.__random_steps = random_steps
        self.__delayed_steps = delayed_steps
        self.__batch_size = batch_size
        self.__context = ctx
        self.__update_step = 0

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

    def __update_model(self):
        self.__update_step += 1
        s, a, s1, r, mask = self.__batch()
        a1 = (self.__actor_target(s1) + self.__noise.sample().clip(*self.__target_noise_clip)).clip(-2.0, 2.0)
        g = r + self.__gamma * mx.nd.min(mx.nd.concat(*[critic(s1, a1) for critic in self.__critic_targets], dim=1), axis=1, keepdims=True) * mask
        with mx.autograd.record():
            for critic in self.__critics:
                L = mx.nd.smooth_l1(mx.nd.abs(g - critic(s, a)))
                L.backward()
        for trainer in self.__critic_trainers:
            trainer.step(self.__batch_size)
        if self.__update_step % self.__delayed_steps == 0:
            with mx.autograd.record():
                L = -self.__critics[0](s, self.__actor(s))
                L.backward()
            self.__actor_trainer.step(self.__batch_size)
            self.__soft_update()

    def __batch(self):
        s, a, s1, r, mask = zip(*random.sample(self.__cache, k=self.__batch_size))
        return mx.nd.concat(*s, dim=0), mx.nd.concat(*a, dim=0), mx.nd.concat(*s1, dim=0), mx.nd.array(r, ctx=self.__context).expand_dims(1), mx.nd.array(mask, ctx=self.__context).expand_dims(1)

    def __soft_update(self):
        for action, target in [(self.__actor, self.__actor_target)] + list(zip(self.__critics, self.__critic_targets)):
            for name, param in target.collect_params().items():
                param.set_data((1 - self.__tau) * param.data(self.__context) + self.__tau * action.collect_params().get(name.removeprefix(target.prefix)).data(self.__context))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of TD3 for Pendulum-v1.")
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
