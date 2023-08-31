#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general

import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from Dotdict import *
import pdb # Use 'pdb.set_trace()' to break

#--------------------------------
# Define Arguments as Dictionary
#--------------------------------
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

#=====================================
# Class' NNET Wrapper
#=====================================
class NNetWrapper():
    #-------------------------
    # Initialization
    #-------------------------
    def __init__(self, game):
        self.nnet = NNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        # Move all model parameters and buffers to GPU
        if args.cuda:
            self.nnet.cuda()

    #-------------------------------------------------
    # Train the Neural Network
    # 'examples' is a list of training data
    # of form (board state, action probability, value)
    #-------------------------------------------------
    def train(self, examples):
        # Learning Optimization Method = Adam
        optimizer = optim.Adam(self.nnet.parameters())
        
        # Repeat Epochs for Train
        for epoch in range(args.epochs):
            print("Train EPOCH ::: %d/%d" % (epoch+1, args.epochs))
            
            # Set the Module in Traing Mode
            self.nnet.train()
            
            # Repeat Batches
            batch_idx = 0
            batch_max = int(len(examples)/args.batch_size)
            l_pi_acc = 0
            l_v_acc  = 0
            while batch_idx < batch_max:
                # Convert Training Example Data for Torch
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))                
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Copy Tensor data in GPU as Contiguous Manner
                if args.cuda:
                    boards     = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs  = target_vs.contiguous().cuda()

                # Compute Output and Loss
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v  = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                
                # Compute Average Loss
                l_pi_acc += l_pi
                l_v_acc  += l_v
                l_pi_ave = l_pi_acc / (batch_idx + 1)
                l_v_ave  = l_v_acc  / (batch_idx + 1)
                
                # Compute Gradient and do SGD (Stochastic Gradient Descent) step
                optimizer.zero_grad() # Clear the Gradient
                total_loss.backward() # Gradient of Tensor
                optimizer.step()      # Single Optimization Step to update parameters
                
                # Increment Batch Index
                batch_idx += 1

                # Print Progress
                print("\rTraining Batch = %d/%d  LossPi = %4f  LossV = %3f" % (batch_idx+1, batch_max, l_pi_ave, l_v_ave), end="")
            print("")

    #-------------------------------------
    # Predict Policy and Value from NN
    #     board : np array of board status
    #-------------------------------------
    def predict(self, board):
        # Preparing Input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        # Set the Module in Evaluation Mode
        self.nnet.eval()
        # Forward Calculation without making backward graph
        with torch.no_grad():
            pi, v = self.nnet(board)
        # Return Result extracted from Tensor's first element
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    #--------------------------------------------
    # Loss Function of Action Probability pi
    #--------------------------------------------
    def loss_pi(self, targets, outputs):
        # Cross Entropy Error
        return -torch.sum(targets * outputs) / targets.size()[0]

    #--------------------------------------------
    # Loss Function of value v
    #--------------------------------------------
    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1))**2) / targets.size()[0]

    #--------------------------------------
    # Save Checkpoint Model Parameters
    #--------------------------------------
    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    #--------------------------------------
    # Load Checkpoint Model Parameters
    #--------------------------------------
    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

#================================================
# Neural Network
#------------------------------------------------
# It takes the raw board state as the input. 
# This is followed by 4 convolutional networks
# and 2 fully connected feedforward networks.
# This is followed by 2 connected layers
# - one that outputs Value v
# - one that outputs Action Probability Vector p
# Training is performed using the Adam optimizer
# with a batch size of 64, with a dropout of 0.3,
# and batch normalisation.
#================================================
class NNet(nn.Module):
    #-----------------------
    # Initialization
    #-----------------------
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    #---------------------
    # Forward Path
    #---------------------
    def forward(self, s):                              # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))            # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))            # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))            # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))            # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        # Use log_softmax to compute error function by multiplication
        return F.log_softmax(pi, dim=1), torch.tanh(v)

#===========================================
# End of Program
#===========================================
