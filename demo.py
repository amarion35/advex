import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from chandelier.models import Model
from chandelier.metrics import sparse_categorical_accuracy
from adversarial_examples import FGSM, GeneralOptimization, LBFGS, DDN

class Classifier(nn.Module):
    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,10)

    def forward(self, x, training):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def demo():

    #  Load datas
    data = load_digits()
    X = data['data']
    Y = data['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.LongTensor(Y_train)
    Y_test = torch.LongTensor(Y_test)

    # Train classifier
    classifier = Classifier(input_shape=64)
    classif_model = Model(classifier, device='cuda:2')
    optimizer = optim.Adam(classifier.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-8)
    loss = nn.CrossEntropyLoss(reduction='mean')
    metrics = [sparse_categorical_accuracy]
    classif_model.compile(optimizer, loss, metrics=metrics)
    classif_model.fit(X_train, Y_train, batch_size=64, epochs=25, validation_data=(X_test, Y_test))

    plot_metrics(classif_model, metrics)

    # Sample data to attack
    idx_sample = np.random.choice(np.arange(X_test.size(0)), 10)
    x_sample = X_test[idx_sample]
    y_sample = Y_test[idx_sample]
    target_adv = torch.empty(x_sample.shape[0], dtype=torch.long).fill_(2)

    # Test attack algorithms
    x_adv = test_fgsm(classif_model, x_sample, target_adv)
    plot_examples(classif_model, x_sample, y_sample, x_adv, target_adv, 'fgsm')
    
    x_adv = test_general(classif_model, x_sample, target_adv)
    plot_examples(classif_model, x_sample, y_sample, x_adv, target_adv, 'general')
    
    x_adv = test_lbfgs(classif_model, x_sample, target_adv)
    plot_examples(classif_model, x_sample, y_sample, x_adv, target_adv, 'lbfgs')
    
    x_adv = test_ddn(classif_model, x_sample, target_adv)
    plot_examples(classif_model, x_sample, y_sample, x_adv, target_adv, 'ddn')


def test_fgsm(model, x_sample, target_adv):
    loss = nn.CrossEntropyLoss(reduce=False)
    algo = FGSM(model, loss, epsilon=1e-1, device='cuda:2')
    x_adv = algo.fit_transform(x_sample, target_adv, iter=50)[-1]
    return x_adv


def test_general(model, x_sample, target_adv):
    loss = nn.CrossEntropyLoss(reduce=False)
    #optimizer = lambda params: optim.LBFGS(params, lr=1e-1, max_iter=20)
    optimizer = lambda params: optim.Adam(params, lr=1e-2)
    algo = GeneralOptimization(model, optimizer, loss, device='cuda:2')
    x_adv = algo.fit_transform(x_sample, target_adv, iter=50)[-1]
    return x_adv

def test_lbfgs(model, x_sample, target_adv):
    loss = nn.CrossEntropyLoss(reduce=False)
    optimizer = lambda params: optim.LBFGS(params, lr=1e-1, max_iter=20)
    algo = LBFGS(model, optimizer, loss, device='cuda:2')
    x_adv = algo.fit_transform(x_sample, target_adv, iter=50)[-1]
    return x_adv

def test_ddn(model, x_sample, target_adv):
    loss = nn.CrossEntropyLoss(reduce=False)
    algo = DDN(model, loss, alpha=2, gamma=0.2, device='cuda:2')
    x_adv = algo.fit_transform(x_sample, target_adv, iter=50)[-1]
    return x_adv

def plot_metrics(model, metrics):
    plt.figure()
    plt.plot(model.hist['loss'], label='loss')
    plt.plot(model.hist['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig('loss')

    for metric in metrics:
        plt.figure()
        plt.plot(model.hist[metric.__name__], label=metric.__name__)
        plt.plot(model.hist['val_'+metric.__name__], label='val_'+metric.__name__)
        plt.legend()
        plt.savefig(metric.__name__)


def plot_examples(model, x_sample, y_sample, x_adv, target_adv, name):
    
    pred_sample = model.predict(x_sample)
    pred_adv = model.predict(x_adv)
    
    x_sample = x_sample.cpu().data.numpy()
    y_sample = y_sample.cpu().data.numpy()
    pred_sample = pred_sample.cpu().data.numpy()
    target_adv = target_adv.cpu().data.numpy()
    x_adv = x_adv.cpu().data.numpy()
    pred_adv = pred_adv.cpu().data.numpy()

    plt.figure(figsize=(15,5))
    for i in range(len(x_sample)):
        plt.subplot(2,10,i+1)
        plt.imshow(x_sample[i].reshape(8,8))
        plt.title('label:{}\npred:{}'.format(y_sample[i], np.argmax(pred_sample[i], axis=-1)))
        plt.axis('off')
        plt.subplot(2,10,i+11)
        plt.imshow(x_adv[i].reshape(8,8))
        plt.title('label:{}\ntarget:{}\npred:{}'.format(y_sample[i], target_adv[i], np.argmax(pred_adv[i], axis=-1)))
        plt.axis('off')
    plt.savefig(name)


def main():
    demo()


if __name__=='__main__':
    main()