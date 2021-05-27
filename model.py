from torch import nn


class DataParallel(nn.DataParallel):
    def __getattr__(self, item):
        if item == 'save_pretrained':
            return getattr(self.module, item)
        else:
            return super().__getattr__(item)