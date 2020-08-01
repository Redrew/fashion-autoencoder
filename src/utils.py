class Config(dict):
    getitem = dict.__getitem__
    setitem = dict.__setitem__
    def __getattr__(self, x):
        try:
            return self.getitem(x.lower())
        except KeyError:
            raise AttributeError(x)
    def __setattr__(self, x, v):
        return self.setitem(x.lower(), v)