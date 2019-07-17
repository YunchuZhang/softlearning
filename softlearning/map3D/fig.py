import os.path as path
import constants as const
import utils_map as utils
import pickle
import types

class Config(object):
    def __init__(self, name, step=0):
        utils.utils.ensure(path.join(const.ckpt_base,const.ckpt_cfg_dir))
        utils.utils.ensure(path.join(const.ckpt_base,const.ckpt_cfg_dir, name.split("/")[0]))
        self.name = path.join(const.ckpt_base,const.ckpt_cfg_dir, name)
        self.snapshot_const(self.name + ".pickle")
        self.dct = {}
        self.step = step

    def snapshot_const(self, file_name):
        with open(file_name, 'wb') as f:
            dump_dict = {key:value for key, value in const.__dict__.items()\
                         if isinstance(value, (int, float, str, bool))}
            pickle.dump(dump_dict, f)

    def save(self):
        with open(self.name, 'w') as f:
            f.write(str(self.step))
            f.write('\n')
            for key, (partscope, partpath) in list(self.dct.items()):
                f.write(key)
                f.write(' : ')
                f.write(partscope)
                f.write(' : ')
                f.write(partpath)
                f.write('\n')

    def load(self):
        with open(self.name, 'r') as f:
            lines = f.readlines()
        self.step = int(lines.pop(0))
        for line in lines:
            partname, partscope, partpath = line.split(':')
            partname = partname.strip()
            partscope = partscope.strip()
            partpath = partpath.strip()
            self.dct[partname] = (partscope, partpath)

    def add(self, partname, partscope, partpath):
        self.dct[partname] = (partscope, partpath)
