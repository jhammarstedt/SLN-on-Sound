import argparse
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data import get_cifar

args = {
    'runs': 5,
    'epochs': 2,
    'stdev': 0.5,
    'lr': 0.001,
    'noise_rate': 0.4,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'batch_size': 128,
    'correction': 250,

    'gpu_id': 0,

    # cifar10
    'sigma': 1 # symmetric
    #sigma: 0.5 # asymmetric

    # cifar100
    #sigma: 0.2
}

device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args['gpu_id'])

net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).cuda()
ema_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).cuda()
for param in ema_net.parameters():
    param.detach_()

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        # copy params
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self): # update weight moving average
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if param.type()=='torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)

optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
ema_optimizer = WeightEMA(net, ema_net)


# Import datasets: Cifar-10, Cifar-100

trainset, testset = get_cifar(dataset='cifar10')
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = trainset.classes

# Add noise to labels


# Train on Wide ResNet-28-2(?)
def train_noise(args, model, device, loader, optimizer, epoch, ema_optimizer):
    net.train()

    return 0, 0

def test(args, model, device, loader, criterion=F.cross_entropy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # pass through model
            test_loss += criterion(output, target, reduction='sum').item() # cross-entropy loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item() # check if correct prediction

    return test_loss/len(loader.dataset), correct/len(loader.dataset) # return average loss and accuracy

total_t0 = time.time()
for epoch in range(1, args['epochs']+1):
    t0 = time.time()

    if epoch > args['correction']: # Label Correction on 250th epoch, without tuning
        args['sigma'] = 0


    _, train_acc = train_noise(args, net, device, train_loader, optimizer, epoch, ema_optimizer)
    _, test_acc = test(args, ema_net, device, test_loader)
    _, test_acc_NoEMA = test(args, net, device, test_loader)
    print('\nEpoch: {} Time: {:.1f}s.\n'.format(epoch, time.time()-t0))


print('\nTotal training time: {:.1f}s.\n'.format(time.time()-total_t0))


net.eval()




# Show results
