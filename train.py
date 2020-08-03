import argparse

from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='')

    return parser.parse_args()


def main(args):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    seed = 8
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # logging.info('base seed {}'.format(seed))
    args = load_json(args.config_path)
    print(args)

    runner = MainRunner(args)

    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
