class AgentBase:
    def __init__(self, environment, test=False):
        self.__environment = environment
        self.__test = test
        self.__state_space = None
        self.__action_space = None

    @property
    def environment(self):
        return self.__environment

    @property
    def test(self):
        return self.__test

    @property
    def spaces(self):
        return self.__state_space, self.__action_space

    @spaces.setter
    def spaces(self, value):
        self.__state_space, self.__action_space = value
