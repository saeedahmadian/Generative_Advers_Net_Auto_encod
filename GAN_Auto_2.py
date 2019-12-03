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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from mpl_toolkits.mplot3d import Axes3D

pca3_mdl = PCA(n_components=3)
pca2_mdl= PCA(n_components=2)
# TSNE
tsne_out = TSNE(n_components=3)

def plot_3d(auto, gen, real, alpha, beta,epoch,batch_step):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.scatter(auto[:, 0], auto[:, 1], auto[:, 2],
                c='r', marker='^',label='Auto-encoder latent variables alpha='+str(alpha)+'beta ='+ str(beta))
    ax2.scatter(gen[:, 0], gen[:, 1], gen[:, 2],
                c='g', marker='*', label='Generator latent variables alpha='+str(alpha)+'beta ='+ str(beta))

    ax3.scatter(real[:, 0], real[:, 1], real[:, 2],
                c='b', marker='+', label='Generator latent variables alpha=' + str(alpha) + 'beta =' + str(beta))
    plt.legend()
    fig.savefig(os.path.join('figs', '{}_{}.png'.format(epoch, batch_step)))
    plt.close(fig)

def plot_2d(auto, gen, real, alpha, beta,epoch,batch_step):
    fig = plt.figure(figsize=[10,15])
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.scatter(auto[:, 0], auto[:, 1],
                c='r', marker='^', label='Auto-encoder latent variables alpha=' + str(alpha) + 'beta =' + str(beta))
    ax2.scatter(gen[:, 0], gen[:, 1],
                c='g', marker='*', label='Generator latent variables alpha=' + str(alpha) + 'beta =' + str(beta))

    ax3.scatter(real[:, 0], real[:, 1],
                c='b', marker='+', label='Generator latent variables alpha=' + str(alpha) + 'beta =' + str(beta))
    plt.legend()
    fig.savefig(os.path.join('figs2d', 'pca_{}_{}.png'.format(epoch, batch_step)))
    plt.close(fig)



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class DiscriminatorNet(nn.Module):
    def __init__(self,num_feature=24,num_z=100,num_out=1):
        super(DiscriminatorNet,self).__init__()
        self.num_feature = num_feature
        self.num_out = num_out
        self.num_z = num_z
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
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(256, self.num_z),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.latent1 = nn.Sequential(
        #     nn.Linear(self.num_z, self.num_z),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        self.hidden2 = nn.Sequential(
            torch.nn.Linear(512, 256),
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
        # x = self.latent1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x



class Auto_Encoder(nn.Module):
    def __init__(self,num_feature=24,num_z=100,num_out=24):
        super(Auto_Encoder,self).__init__()
        self.num_feature = num_feature
        self.num_out = num_out
        self.num_z= num_z
        self.hidden0 = nn.Sequential(
            nn.Linear(self.num_feature, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.encoder = nn.Sequential(
            nn.Linear(256, self.num_z),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(self.num_z, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(64, self.num_out)

        )

    def forward(self, x):
        x = self.hidden0(x)
        latent = self.encoder(x)
        x1 = self.hidden1(latent)
        decoder = self.decoder(x1)
        return latent, decoder



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
autoencoder = Auto_Encoder()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()
    autoencoder.cuda()


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
auto_optimizer= optim.Adam(autoencoder.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

#loss2

loss_auto = nn.SmoothL1Loss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 200


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

def train_auto(optimizer, data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Desired Data
    _,prediction = autoencoder.forward(data)
    # Calculate error and backpropagate
    error = loss_auto(prediction, data)
    error.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error, prediction,optimizer

def train_generator(optimizer, fake_data,desired_data,alpha,beta):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    latent, auto_pred = autoencoder.forward(desired_data)
    # Sample noise and generate fake data
    prediction_d = discriminator(fake_data)

    prediction_g= generator.forward(noise(real_data.size(0))+latent)
    # Calculate error and backpropagate
    error_gan = loss(prediction_d, real_data_target(prediction_d.size(0)))
    err_auto = loss_auto(prediction_g,auto_pred)
    error = beta*error_gan + alpha* err_auto
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error_gan, err_auto,error,optimizer




# des = np.load('desired_data.npy')
# reg = np.load('regular_data.npy')
# scale= StandardScaler((0,1))
# des_scale= scale.fit(des).transform(des)
# reg_scale= scale.fit(reg).transform(reg)
# x_train_des, x_test_des, _,_ = train_test_split(des_scale,np.ones((des_scale.shape[0],1)),test_size=.3)
# x_train_reg, x_test_reg, _,_ = train_test_split(reg_scale,np.ones((reg_scale.shape[0],1)),test_size=.3)

x_train_des = np.load('x_train_des.npy')
x_test_des = np.load('x_test_des.npy')
x_train_reg = np.load('x_train_reg.npy')
x_test_reg = np.load('x_test_reg.npy')

# reg = np.load('regular_data.npy')
# Create loader with data, so that we can iterate over it
batch_size=20
# data_loader = torch.utils.data.DataLoader(des, batch_size=batch_size, shuffle=True)
# Num batches
num_batches = int(x_train_des.shape[0]/batch_size)-1


def randomize(data):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation, :]
    # shuffled_y = y[permutation]
    return shuffled_data

# logger = Log(model_name='VGAN', data_name='MNIST')

total_g_err = []
total_g_gan_err=[]
total_g_auto_err=[]
total_d_err = []
total_d_err_real = []
total_d_err_fake = []
total_auto_err=[]


d_optimizer.zero_grad()
g_optimizer.zero_grad()

data_training= np.concatenate([x_train_des,x_train_reg], axis=0)
alpha = .2
beta =.8

for epoch in range(num_epochs):

    x_train_disc= randomize(data_training)
    x_train_auto = randomize(x_train_des)
    # for real_batch,_ in data_loader:
    for step in range(num_batches):
        # 1. Train Discriminator
        data_batch_disc = x_train_disc[step*batch_size:(1+step)*batch_size,:]
        data_batch_auto = x_train_auto[step*batch_size:(1+step)*batch_size,:]
        real_data = torch.tensor(data_batch_disc).float().to(device)
        # real_data= Variable(torch.tensor(data_batch_disc))
        auto_data = torch.tensor(data_batch_auto).float().to(device)
        # auto_data= Variable(torch.tensor(data_batch_auto))
            # Variable(images_to_vectors(real_batch))
        # if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        auto_latent, auto_pred = autoencoder.forward(auto_data)
        fake_data= generator.forward(noise(real_data.size(0))+auto_latent).detach()
        # fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        new_fake_data = beta*fake_data + alpha*auto_pred

        d_error_real, d_error_fake, d_pred_real, d_pred_fake,d_optimizer = train_discriminator(d_optimizer,
                                                                real_data, new_fake_data)

        # train auto-encoder
        auto_err, auto_output, auto_optimizer = train_auto(auto_optimizer,auto_data)

        # 2. Train Generator
        # Generate fake data
        auto_latent, auto_pred = autoencoder.forward(auto_data)
        fake_data = generator(noise(data_batch_auto.shape[0])+auto_latent)

        new_fake_data = beta * fake_data + alpha * auto_pred
        # Train G
        g_gan,g_auto,g_error,g_optimizer = train_generator(g_optimizer, new_fake_data,auto_data,alpha,beta)
        # Log error
        # logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress
        if (step) % 10 == 0:
            print('___________________________________________________')
            print('epoch {}, the G Loss is : {} '.format(epoch,g_error))
            print ('epoch {}, the D Loss is : {} '.format(epoch,d_error_fake+d_error_real))
            print('epoch {}, the Auto Loss is : {} '.format(epoch, auto_err))
            print('___________________________________________________')
            total_g_err.append(g_error)
            total_g_gan_err.append(g_gan)
            total_g_auto_err.append(g_auto)
            total_d_err.append(d_error_fake+d_error_real)
            total_d_err_fake.append(d_error_fake)
            total_d_err_real.append(d_error_real)
            total_auto_err.append(auto_err)
    pca3_gen = pca3_mdl.fit_transform(new_fake_data.detach().numpy())
    pca3_auto = pca3_mdl.fit_transform(auto_pred.detach().numpy())
    pca3_des = pca3_mdl.fit_transform(data_batch_auto)
    pca2_gen = pca2_mdl.fit_transform(new_fake_data.detach().numpy())
    pca2_auto = pca2_mdl.fit_transform(auto_pred.detach().numpy())
    pca2_des = pca2_mdl.fit_transform(data_batch_auto)
    plot_3d(pca3_auto, pca3_gen, pca3_des, alpha, beta, epoch, step)
    plot_2d(pca2_auto, pca2_gen, pca2_des, alpha, beta, epoch, step)
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

torch.save(generator,'generator_net_hybrid.pt')
torch.save(discriminator,'discriminator_net_hybrid.pt')
torch.save(autoencoder,'autoencoder_hybrid.pt')


batch_test = 5
num_batch_test = int(x_test_des.shape[0]/batch_test)
num_epochs_test = 10
test_mode = False
N_out = 1
softmax = nn.Sequential(nn.Softmax(dim=N_out))
if test_mode:
    generator = torch.load('generator_net_hybrid.pt')
    discriminator_hybrid = torch.load('discriminator_net_simple_GAN.pt')
    autoencoder = torch.load('autoencoder_hybrid.pt')
    discriminator = torch.load('discriminator_net_hybrid.pt')

    prec_real_simple = []
    prec_real_hybrid = []
    prec_fake_simple = []
    prec_fake_hybrid = []
    err_real_simple = []
    err_real_hybrid = []
    err_fake_simple = []
    err_fake_hybrid = []
    for epoch_test in range(num_epochs_test):
        x_test_disc = randomize(x_test_reg)
        x_test_auto = randomize(x_test_reg)
        for step in range(num_batch_test):
            data_batch_disc = x_test_disc[step * batch_test:(1 + step) * batch_test, :]
            data_batch_auto = x_test_auto[step * batch_test:(1 + step) * batch_test, :]
            real_data = torch.tensor(data_batch_disc).float().to(device)
            auto_data =  torch.tensor(data_batch_auto).float().to(device)
            with torch.no_grad():
                auto_latent, auto_pred = autoencoder.forward(auto_data)
                fake_data = generator.forward(noise(real_data.size(0)) + auto_latent).detach()

                fake_data = beta*fake_data+ alpha*auto_pred
                prediction_real = discriminator(real_data)
                prediction_real_hybrid = discriminator_hybrid(real_data)
                c_simple=0
                c_hybrid=0
                for i in range(batch_test):
                    if prediction_real.numpy()[i] >.5:
                        c_simple=c_simple+1
                    if prediction_real_hybrid.numpy()[i]> 0.5:
                        c_hybrid =c_hybrid+1
                prec_real_simple.append(c_simple / batch_test)
                prec_real_hybrid.append(c_hybrid / batch_test)
                # Calculate error and backpropagate
                error_r_simple = loss(prediction_real, real_data_target(real_data.size(0)))
                error_r_hybrid = loss(prediction_real, real_data_target(real_data.size(0)))
                err_real_simple.append(error_r_simple)
                err_real_hybrid.append(error_r_hybrid)

                prediction_fake_simple = discriminator(fake_data)
                prediction_fake_hybrid = discriminator_hybrid(fake_data)
                c_simple = 0
                c_hybrid = 0
                for i in range(batch_test):
                    if prediction_fake_simple.numpy()[i] < .5:
                        c_simple = c_simple + 1
                    if prediction_fake_hybrid.numpy()[i] < .5:
                        c_hybrid = c_hybrid + 1
                prec_fake_simple.append(c_simple / batch_test)
                prec_fake_hybrid.append(c_hybrid/batch_test)
                # Calculate error and backpropagate
                error_f_simple = loss(prediction_fake_simple, fake_data_target(real_data.size(0)))
                error_f_hybrid = loss(prediction_fake_hybrid, fake_data_target(real_data.size(0)))
                err_fake_simple.append(error_f_simple)
                err_fake_hybrid.append(error_f_hybrid)








a=1