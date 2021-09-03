import os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
#import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import grad
from torch.autograd import Variable
#import tensorboardX
import checkpoint
from FaceLandmarksDataset import FaceLandmarksDataset

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# the folder of results
name = 'train_1_5CNN'

out_folder = './marknet_training/'
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)

out_folder = out_folder + name
if not os.path.isdir(out_folder):
    os.mkdir(out_folder)


##############################
#######  NETWORK PART  #######
##############################

d=64
c=1
rad = 2
mark_num = 68

class marknet(nn.Module):
    # initializers
    def __init__(self):
        super(marknet, self).__init__()
        self.conv1 = nn.Conv2d(1, d, (4,4), (2,2), (1,1))
        self.conv2 = nn.Conv2d(d, d*2, (4,4), (2,2), (1,1))
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, (4,4), (2,2), (1,1))
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, (4,4), (2,2), (1,1))
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, d*16, (4,4), (2,2), (1,1))

        self.linear = nn.Linear(d*64, 2*mark_num)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = x.reshape(x.size(0), d*64)
        x = self.linear(x)

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


##############################
########   UTILITIES  ########
##############################

def show_train_hist(hist, show = False, save = False, path = 'train_hist.png'):
    x = range(len(hist['train_losses']))

    y1 = hist['train_losses']
    y2 = hist['val_losses']

    # Discriminator
    plt.close('all')

    plt.plot(x, y1, label='train_loss')
    plt.plot(x, y2, label='val_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_marks(marks, show = False, save = False, path = 'Val_marks.png'):
    x = marks.cpu().numpy()[0,:,0]

    y = marks.cpu().numpy()[0,:,1]
    # Discriminator
    plt.close('all')

    plt.plot(x, y, linestyle = 'dotted', marker = '.', label='detected landmarks')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()



##############################
##########  SETTING ##########
##############################

batch_size = 64
lr = 0.0001
train_epoch = 25300

# data_loader
trainset = FaceLandmarksDataset(usage = 'train')
trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True, drop_last= True)

valset = FaceLandmarksDataset(usage = 'val')
valloader = torch.utils.data.DataLoader(dataset = valset, batch_size = batch_size, shuffle = False, drop_last= True)


# network
D = marknet()
D.weight_init(mean=0.0, std=0.02)
D = D.to(device)

#criterion
loss = nn.MSELoss()

# Adam optimizer
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

""" load checkpoint """
ckpt_dir = './marknet_chekpoints/'
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

ckpt_dir = ckpt_dir + name
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

try:
    ckpt = checkpoint.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    min_val_loss = ckpt['min_val_loss']
    train_hist = ckpt['train_hist']
    D.load_state_dict(ckpt['D'])
    D_optimizer.load_state_dict(ckpt['d_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0
    min_val_loss = 100
    train_hist = {}
    train_hist['train_losses'] = []
    train_hist['val_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

##############################
#######  TRAINING PART #######
##############################

#writer = tensorboardX.SummaryWriter(out_folder + '/summaries/')

print('training start!')
start_time = time.time()

is_best = True

for epoch in range(start_epoch, train_epoch):

    train_losses = []
    val_losses = []

    epoch_start_time = time.time()
    for i, (face, mark) in enumerate(trainloader):
        face = face.type(torch.cuda.FloatTensor).to(device)
        mark = mark.type(torch.cuda.FloatTensor).to(device)
        step = epoch * len(trainloader) + i + 1
        #print(step)
        #print(x_.shape)

        # train marknet D
        D.zero_grad()

        output = D(face)

        D_train_loss = loss(output, mark)

        D_train_loss.backward()
        D_optimizer.step()

        train_losses.append(D_train_loss.data)

        #writer.add_scalar('Train_loss', D_train_loss.data.cpu().numpy(), global_step=step)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    D.eval()

    with torch.no_grad():

        for i, (face, mark) in enumerate(valloader):
            face = face.type(torch.cuda.FloatTensor).to(device)
            mark = mark.type(torch.cuda.FloatTensor).to(device)
            #step = epoch * len(trainloader) + i + 1
            #print(step)
            #print(x_.shape)

            output = D(face)

            if ((epoch + 1) % 10 == 0) and (i==0):
                p = out_folder + '/markplots/'

                if not os.path.isdir(p):
                    os.mkdir(p)

                show_marks(output.reshape(output.size(0), 68, 2), show = False, save = True, path = p + 'epoch_' + str(epoch + 1)+'.png')

            D_val_loss = loss(output, mark)

            val_losses.append(D_val_loss.data)

            #writer.add_scalar('Val_loss', D_val_loss.data.cpu().numpy(), global_step=step)

    D.train()

    print('[%d/%d] - ptime: %.2f, train_loss: %.3f, val_loss: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(train_losses))*1000000,
    torch.mean(torch.FloatTensor(val_losses))*1000000))
    
    if min_val_loss>torch.mean(torch.FloatTensor(val_losses)):
        min_val_loss = torch.mean(torch.FloatTensor(val_losses))
        is_best = True
        print('*** best epoch: %d' % (epoch + 1))
    else:
        is_best = False

    train_hist['train_losses'].append(torch.mean(torch.FloatTensor(train_losses)))
    train_hist['val_losses'].append(torch.mean(torch.FloatTensor(val_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    show_train_hist(train_hist, show=False, save=True, path=out_folder + '/marknet_train_hist.png')

    checkpoint.save_checkpoint({'epoch': epoch + 1,
                           'min_val_loss': min_val_loss,
    					             'train_hist': train_hist,
                           'D': D.state_dict(),
                           'd_optimizer': D_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1), is_best = is_best,
                          max_keep=2)


end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(D.state_dict(), out_folder + '/marknet_param.pkl')
torch.save(D_optimizer.state_dict(), out_folder + '/optimizer_param.pkl')

with open(out_folder + '/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=out_folder + '/marknet_train_hist.png')

'''
images = []
for epoch in range(train_epoch):
    if((epoch+1)%100 == 0):
        img_name = out_folder + '/Fixed_results/epoch_' + str(epoch + 1) + '.png'
        images.append(imageio.imread(img_name))
imageio.mimsave(out_folder + '/generation_animation.gif', images, fps=5)
'''
