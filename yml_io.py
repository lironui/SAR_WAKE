import yaml
import os


class YamlHandler(object):
    def __init__(self, filename):
        self.filename = filename

    def read_yaml(self):
        """read yaml files"""
        with open(self.filename, encoding='utf-8') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    def write_yaml(self, data, encoding='utf-8'):
        """write yaml files"""
        with open(self.filename, encoding=encoding, mode='w') as f:
            return yaml.dump(data, stream=f, allow_unicode=True)
