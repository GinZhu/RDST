from torch import optim
import torch.optim.lr_scheduler as lrs
import time


class Timer(object):
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.opt == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.opt == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.opt == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    else:
        raise ValueError('Optimizer mush be [SGD, Adam, RMSprop], but got {}'.format(
            args.epsilon
        ))

    kwargs['lr'] = args.learning_rate
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if 'step' in args.lr_decay_type:
        step_size = int(args.lr_decay_type.split()[-1])
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=step_size,
            gamma=args.lr_decay_gamma
        )
    elif args.lr_decay_type.find('milestones') >= 0:
        milestones = args.lr_decay_type.split(' ')[1:]
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.lr_decay_gamma
        )
    else:
        scheduler = None

    return scheduler
