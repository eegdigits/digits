class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def getoutname(config):
    return gettypename(config) + "_{}".format(config.subject)

def gettypename(config):
    dconfig = dict(config)
    name = ''
    for step in config.pipeline.steps:
        name += step[0]

    name += '_ss{}'.format(config.subsample_width)
    name += '_{}_{}'.format(config.t0, config.t1)
    if config.fft:
        name += '_fft'
        try:
            name += '_{}'.format(dconfig['fmin'])
        except KeyError:
            name += '_min'
        try:
            name += '_{}'.format(dconfig['fmax'])
        except KeyError:
            name += '_max'
        try:
            name += '_{}bins'.format(dconfig['size'])
        except KeyError:
            pass
        try:
            power = dconfig['power']
            if power:
                name += '_power'
            else:
                name += '_nopower'
        except KeyError:
            name += '_nopower'
    else:
        name += '_raw'

    return name
