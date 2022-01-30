import json

class WeightExponentialMovingAverage:
    def __init__(self, model, momentum_model, alpha=0.999):
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.momentum_params = list(momentum_model.state_dict().values())

    def step(self):
        for param, momentum_param in zip(self.params, self.momentum_params):
            momentum_param.copy_(momentum_param.data*self.alpha + param.data*(1.0 - self.alpha))

class TrainingLogger:
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.test_loss_NoEMA = []
        self.test_acc_NoEMA = []

    def save_epoch(self, train_loss, train_acc, test_loss, test_acc, test_loss_NoEMA, test_acc_NoEMA):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)
        self.test_loss_NoEMA.append(test_loss_NoEMA)
        self.test_acc_NoEMA.append(test_acc_NoEMA)

    def print_last_epoch(self, logger=None):
        log_string = 'Train loss:\t{:.3f}\tTest loss:\t{:.3f}\tTest loss NoEMA:\t{:.3f}\t'.format(
            self.train_loss[-1], self.test_loss[-1], self.test_loss_NoEMA[-1])
        if not logger:
            print(log_string)
        else:
            logger.info(log_string)

    def export_as_json(self, path):
        data = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'test_loss': self.test_loss,
            'test_acc': self.test_acc,
            'test_loss_NoEMA': self.test_loss_NoEMA,
            'test_acc_NoEMA': self.test_acc_NoEMA
        }
        with open(path, 'w') as f:
            json.dump(data, f)
