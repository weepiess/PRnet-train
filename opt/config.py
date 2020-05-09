import argparse
import tensorflow as tf


class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, par):
        par.add_argument('--learning_rate', default=0.0002, type=float, help='The learning rate')
        par.add_argument('--epochs', default=100, type=int, help='Total epochs')
        par.add_argument('--batch_size', default=16, type=int, help='Batch sizes')
        par.add_argument('--gpu', default='0', type=str, help='The GPU ID')
        self.initialized = True
        return par

    def get_config(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        config, _ = parser.parse_known_args()

        return config