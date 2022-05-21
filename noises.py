import mxnet as mx


class OrnsteinUhlenbeckNoise:
    def __init__(self, shape, mu=0.0, theta=1.0, sigma=0.2, ctx=mx.cpu()):
        self.__state = mx.nd.ones(shape, ctx=ctx) * mu
        self.__mu = mu
        self.__theta = theta
        self.__sigma = sigma

    def sample(self):
        self.__state += (self.__mu - self.__state) * self.__theta + mx.nd.random.uniform_like(self.__state, high=self.__sigma)
        return self.__state

    def reset(self):
        self.__state = mx.nd.ones_like(self.__state) * self.__mu
