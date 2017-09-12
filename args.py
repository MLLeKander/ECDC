import argparse

class CustomParser(argparse.ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if 'help' not in kwargs:
            kwargs['help'] = '(default: %(default)s)'
        return argparse.ArgumentParser.add_argument(self, *args, **kwargs)

arg_parser = CustomParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
