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

    @property
    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + mx.nd.log(self.__std)

    def sample(self):
        return mx.nd.random.normal(self.__mu, self.__std)

    def rsample(self):
        return self.__mu + mx.nd.random.normal_like(self.__std) * self.__std

    def probability(self, x):
        return 1 / (self.__std * math.sqrt(2 * math.pi)) * mx.nd.exp(-((x - self.__mu) ** 2 / (2 * self.__std ** 2)))

    def log_prob(self, x):
        return -((x - self.__mu) ** 2) / (2 * self.__std ** 2) - mx.nd.log(self.__std) - math.log(math.sqrt(2 * math.pi))
