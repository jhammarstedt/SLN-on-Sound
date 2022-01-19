
class WeightExponentialMovingAverage:
    def __init__(self, model, ema_model, alpha=0.999):
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if param.type() != 'torch.cuda.LongTensor':
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
