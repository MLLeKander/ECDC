import argparse
import functools
import inspect

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_action(self, action):
        if not action.help:
            action.help = '(default: %(default)s)'
        return super(CustomFormatter, self)._format_action(action)

arg_parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
args = argparse.Namespace()

def parse_args():
    global args
    args = arg_parser.parse_args(namespace=args)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(ctor):
    def f(arg):
        try:
            return map(ctor, arg.split(','))
        except Exception, e:
            raise argparse.ArgumentTypeError('Invalid element in list: %s'%e)
    return f

CLIArg = object()

def clidefault(func):
    argspec = inspect.getargspec(func)

    @functools.wraps(func)
    def wrapper_clidefault(*posargs, **kwargs):
        if argspec.defaults is not None:
            #TODO: Error messages could be better here (ie when posargs is too short)
            for i in range(-1, -min(len(argspec.args)-len(posargs),len(argspec.defaults))-1, -1):
            #for i in range(-1, -len(argspec.defaults)-1, -1):
                default = argspec.defaults[i]
                arg = argspec.args[i]
                if default is CLIArg and (arg not in kwargs or kwargs[arg] is CLIArg):
                    if arg not in args:
                        raise NameError('Argument "%s" not found in args'%arg)
                    kwargs[arg] = vars(args)[arg]
        return func(*posargs, **kwargs)
    return wrapper_clidefault
