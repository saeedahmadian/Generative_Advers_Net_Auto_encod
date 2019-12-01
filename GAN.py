from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import torch.nn as nn
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.model_selection import train_test_split
# from utils import Logger as Log

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class DiscriminatorNet(nn.Module):
    def __init__(self,num_feature=24,num_out=1):
        super(DiscriminatorNet,self).__init__()
        self.num_feature = num_feature
        self.num_out = num_out
        self.hidden0 = nn.Sequential(
            nn.Linear(self.num_feature, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, self.num_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class GeneratorNet(nn.Module):
    def __init__(self,num_z=100,num_out=24):
        super(GeneratorNet,self).__init__()
        self.num_z = num_z
        self.num_out = num_out

        self.hidden0 = nn.Sequential(
            nn.Linear(self.num_z, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, self.num_out)
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


discriminator = DiscriminatorNet()
generator = GeneratorNet()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 0


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

num_test_samples = 16
test_noise = noise(num_test_samples)


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator.forward(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator.forward(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real , error_fake, prediction_real, prediction_fake,optimizer


def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error,optimizer


des = np.load('desired_data.npy')
x_train, x_test, _,_ = train_test_split(des,np.ones((des.shape[0],1)),test_size=.3)

# reg = np.load('regular_data.npy')
# Create loader with data, so that we can iterate over it
batch_size=20
data_loader = torch.utils.data.DataLoader(des, batch_size=batch_size, shuffle=True)
# Num batches
num_batches = int(x_train.shape[0]/batch_size)-1


def randomize(data):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation, :]
    # shuffled_y = y[permutation]
    return shuffled_data

# logger = Log(model_name='VGAN', data_name='MNIST')

total_g_err = []
total_d_err = []
total_d_err_real = []
total_d_err_fake = []

d_optimizer.zero_grad()
g_optimizer.zero_grad()

for epoch in range(num_epochs):

    x_train= randomize(x_train)

    # for real_batch,_ in data_loader:
    for step in range(num_batches):
        # 1. Train Discriminator
        data_batch = x_train[step*batch_size:(1+step)*batch_size,:]
        real_data = torch.tensor(data_batch).float().to(device)
            # Variable(images_to_vectors(real_batch))
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data= generator.forward(noise(real_data.size(0))).detach()
        # fake_data = generator(noise(real_data.size(0))).detach()
        # Train D

        d_error_real, d_error_fake, d_pred_real, d_pred_fake,d_optimizer = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(len(data_batch)))
        # Train G
        g_error,g_optimizer = train_generator(g_optimizer, fake_data)
        # Log error
        # logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress
        if (step) % 10 == 0:
            print('___________________________________________________')
            print('epoch {}, the G Loss is : {} '.format(epoch,g_error))

            print ('epoch {}, the D Loss is : {} '.format(epoch,d_error_fake+d_error_real))
            print('___________________________________________________')
            total_g_err.append(g_error)
            total_d_err.append(d_error_fake+d_error_real)
            total_d_err_fake.append(d_error_fake)
            total_d_err_real.append(d_error_real)
            # display.clear_output(True)
            # # Display Images
            # test_images = vectors_to_images(generator(test_noise)).data.cpu()
            # logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
            # # Display status Logs
            # logger.display_status(
            #     epoch, num_epochs, n_batch, num_batches,
            #     d_error, g_error, d_pred_real, d_pred_fake
            # )
        # Model Checkpoints
        # logger.save_models(generator, discriminator, epoch)

torch.save(generator,'generator_net.pt')
torch.save(discriminator,'discriminator_net.pt')

batch_test = 5
num_batch_test = int(x_test.shape[0]/batch_test)
num_epochs_test = 1
test_mode = True
N_out = 1
softmax = nn.Sequential(nn.Softmax(dim=N_out))
if test_mode:
    generator = torch.load('generator_net.pt')
    discriminator = torch.load('discriminator_net.pt')
    prec_real=[]
    prec_fake=[]
    err_real_test=[]
    err_fake_test=[]
    for epoch_test in range(num_epochs_test):
        x_test = randomize(x_test)
        for step in range(num_batch_test):
            data_batch = x_test[step * batch_test:(1 + step) * batch_test, :]
            test_data = torch.tensor(data_batch).float().to(device)
            with torch.no_grad():
                fake_data = generator.forward(noise(test_data.size(0))).detach()
                prediction_real = discriminator(test_data)
                c=0
                for i in prediction_real.numpy():
                    if i >.5:
                        c=c+1
                prec_real.append(c/test_data.size(0))
                # Calculate error and backpropagate
                error_real = loss(prediction_real, real_data_target(test_data.size(0)))
                err_real_test.append(error_real)

                prediction_fake = discriminator(fake_data)
                c = 0
                for i in prediction_fake.numpy():
                    if i < .5:
                        c = c + 1
                prec_fake.append(c / test_data.size(0))
                # Calculate error and backpropagate
                error_fake = loss(prediction_fake, fake_data_target(test_data.size(0)))
                err_fake_test.append(error_fake)








a=1