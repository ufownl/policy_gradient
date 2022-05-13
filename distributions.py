import math
import mxnet as mx


class Normal:
    def __init__(self, mu, std):
        self.__mu = mu
        self.__std = std

    @property
    def mean(self):
        return self.__mu

    @property
    def standard_deviation(self):
        return self.__std

    def sample(self):
        return mx.nd.random.normal(self.__mu, self.__std)

    def probability(self, x):
        return 1 / (self.__std * math.sqrt(2 * math.pi)) * mx.nd.exp(-((x - self.__mu) ** 2 / (2 * self.__std ** 2)))
