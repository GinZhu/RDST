import configparser
from os.path import isdir, join, exists
from os import makedirs


class ParametersLoader(object):

    def __init__(self, config_file):

        # ## loading parameters
        config = configparser.ConfigParser()
        config.read(config_file)

        self.names = []

        for s in config.keys():
            for k in config[s].keys():
                if k in self.names:
                    continue
                self.names.append(k)
                setattr(self, k, eval(config.get(s, k)))

    def __str__(self):
        plog = 'Parameters: \n'
        for k in self.names:
            plog += '  {}: \t {} \n'.format(
                k, self.__getattribute__(k)
            )
        return plog

