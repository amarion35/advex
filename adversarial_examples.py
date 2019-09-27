import torch
from torch.autograd import Variable
import torch.optim as optim

class FGSM:

    def __init__(self, model, loss, epsilon=0.001, device='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.epsilon = epsilon
        self.loss_function = loss

    def fit_transform(self, x, target, iter=1, early_stopping=True):
        xadv = []
        xadv.append(x.clone().to(self.device))
        target = target.to(self.device)
        pred = self.model.predict_batch(xadv[-1])
        print('Fit FGSM')
        print('loss: {:.4f}'.format(self.loss_function(pred, target).mean()), end='')
        grad = self._x_grad(xadv[-1], target)
        print(' - grad: {:.4f}'.format(grad.norm().cpu().data.numpy()), end='')
        print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()), end='')
        print(' - target: {}'.format(target.cpu().data.numpy()))
        for i in range(iter):
            print('Iter {}'.format(i), end='')
            grad = self._x_grad(xadv[-1], target)
            print(' - grad: {:.4f}'.format(grad.norm().cpu().data.numpy()), end='')
            xadv.append(xadv[-1]-self.epsilon*torch.sign(grad))
            pred = self.model.predict_batch(xadv[-1])
            print(' - loss: {:.4f}'.format(self.loss_function(pred, target).mean()), end='')
            print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()))
            if early_stopping and torch.equal(torch.argmax(pred, dim=1), target):
                print('Early Stopping: Example found')
                break
        return xadv

    def _x_grad(self, x, target):
        x = Variable(x, requires_grad=True)
        out = self.model.predict_batch(x)
        loss = self.loss_function(out, target)
        [l.backward(retain_graph=True) for l in loss]
        g = x.grad
        assert bool(torch.isnan(g).any())==False
        return g
        
class GeneralOptimization:

    def __init__(self, model, optimizer, loss, device='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.loss_function = loss
        self.optimizer = optimizer

    def fit_transform(self, x, target, iter=1, early_stopping=True):
        x = x.clone().to(self.device)
        x.requires_grad = True
        optimizer = self.optimizer([x])
        xadv = []
        xadv.append(x.clone().to(self.device))
        target = target.to(self.device)
        pred = self.model.predict_batch(xadv[-1])

        print('Fit GeneralOptimization')
        print('loss: {:.4f}'.format(self.loss_function(pred, target).mean()), end='')
        print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()), end='')
        print(' - target: {}'.format(target.cpu().data.numpy()))

        def closure():
            optimizer.zero_grad()
            out = self.model.predict_batch(x)
            loss = self.loss_function(out, target)
            for l in loss:
                l.backward(retain_graph=True)
            for l in loss:
                yield l

        def closure_wrapper():
            return next(closure())

        for i in range(iter):
            print('Iter {}'.format(i), end='')
            for _ in range(x.size(0)):
                optimizer.step(closure_wrapper)

            xadv.append(x.clone().to(self.device))
            pred = self.model.predict_batch(xadv[-1])
            print(' - loss: {:.4f}'.format(self.loss_function(pred, target).mean()), end='')
            print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()))

            if early_stopping and torch.equal(torch.argmax(pred, dim=1), target):
                print('Early Stopping: Example found')
                break
        return xadv

class LBFGS:

    def __init__(self, model, optimizer, loss, device='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.loss_function = lambda n, x, xadv, out, y: n*(xadv-x).norm() + loss(out, y)
        self.optimizer = optimizer

    def fit_transform(self, x, target, iter=1):
        x = x.clone().to(self.device)
        x.requires_grad = True
        optimizer = self.optimizer([x])

        n = torch.tensor(0.1).to(self.device)
        n_optimizer = optim.SGD([n], lr=0.1)

        xadv = []
        xadv.append(x.clone().to(self.device))
        target = target.to(self.device)
        pred = self.model.predict_batch(xadv[-1])

        print('Fit LBFGS')
        print('loss: {:.4f}'.format(self.loss_function(n, xadv[0], xadv[-1], pred, target).mean()), end='')
        print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()), end='')
        print(' - target: {}'.format(target.cpu().data.numpy()))

        def closure():
            optimizer.zero_grad()
            out = self.model.predict_batch(x)
            loss = self.loss_function(n, xadv[0], x, out, target)
            for l in loss:
                l.backward(retain_graph=True)
            for l in loss:
                yield l

        def closure_wrapper():
            return next(closure())

        for i in range(iter):
            print('Iter {}'.format(i), end='')
            for _ in range(x.size(0)):
                optimizer.step(closure_wrapper)
                n_optimizer.step()

            xadv.append(x.clone().to(self.device))
            pred = self.model.predict_batch(xadv[-1])
            print(' - loss: {:.4f}'.format(self.loss_function(n, xadv[0], xadv[-1], pred, target).mean()), end='')
            print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()))
        return xadv


class DDN:
    
    def __init__(self, model, loss, alpha=0.08, gamma=1, device='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.loss_function = loss

    def fit_transform(self, x, target, iter, verbose=1):
        x = x.clone().to(self.device)
        xadv = []
        xadv.append(x.clone().to(self.device))
        target = target.to(self.device)
        pred = self.model.predict_batch(xadv[-1])

        print('Fit DDN')
        print('loss: {:.4f}'.format(self.loss_function(pred, target).mean()), end='')
        print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().data.numpy()), end='')
        print(' - target: {}'.format(target.cpu().data.numpy()))

        delta = torch.empty(x.size()).fill_(0).to(self.device)
        epsilon = torch.empty(x.size(0)).fill_(1).to(self.device)
        m = torch.empty(x.size(0)).fill_(-1).to(self.device)
        alpha = torch.empty(x.size(0)).fill_(self.alpha).to(self.device)
        gamma = torch.empty(x.size(0)).fill_(self.gamma).to(self.device)

        for i in range(iter):
            print('Iter {}'.format(i), end='')
            g = torch.einsum('i,ij->ij' , m, self._grad(xadv[-1], target))
            g = torch.einsum('i,ij->ij' , alpha, (g/(g.norm()+0.0001)))
            delta = delta+g
            sign = torch.eq(self._vote(pred), target)
            sign = sign.float()
            sign = sign*(-2)+1
            sign = sign*gamma
            epsilon = (1+sign)*epsilon
            norm_delta = torch.div(delta.transpose(0,1), delta.norm(dim=1)).transpose(0,1)
            step = torch.einsum('ij,i->ij', norm_delta, epsilon)

            loss = self.loss_function(pred, target)

            xadv.append(x+step)
            pred = self.model.predict_batch(xadv[-1])
            print(pred[0].data.numpy())
            print(' - loss: {:.4f}'.format(self.loss_function(pred, target).mean()), end='')
            print(' - class: {}'.format(torch.argmax(pred, dim=1).cpu().numpy()))
        return xadv

    def _grad(self, x, target):
        x = Variable(x, requires_grad=True)
        out = self.model.predict_batch(x)
        loss = self.loss_function(out, target)
        [l.backward(retain_graph=True) for l in loss]
        g = x.grad
        assert bool(torch.isnan(g).any())==False
        return g

    def _vote(self, out):
        return torch.argmax(out, dim=1)