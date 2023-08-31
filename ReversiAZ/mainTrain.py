#!/usr/bin/python3
#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general

import numpy as np
import os, sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from Match import Match
from MCTS import MCTS
from Game import ReversiGame
from NNet import NNetWrapper as nn
from Dotdict import *
import pdb # Use 'pdb.set_trace()' to break

#--------------------------------
# Define Arguments as Dictionary
#--------------------------------
args = dotdict({
    'boardSize': 6,
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'matchCompare': 40,
    'cpuct': 1,

    'checkpoint': './model/',
    'load_model': False,
    'load_folder_file': ('./model','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    
    'file_folder': './model/',
    'file_model_best': 'best_model.tar',
    'file_model_temp': 'temp_model.tar',
    'file_model': '_model.tar',
    'file_train': '_train.examples',
    'file_iter' : 'iter.txt',
})

#=======================================
# Class: Teach
#     Executes Self play + Learning. 
#=======================================
class Teach():
    #------------------
    # Initialization
    #------------------
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
            # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
            # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False
            # can be overriden in loadTrainExamples()

    #--------------------------------------------------------------
    # Executes one episode of self-play, starting with player1. 
    #--------------------------------------------------------------
    # Each turn is added as a training example to trainExamples.
    # After the game ends, the outcome of the game is used
    # to assign values to each exampl in trainExamples.
    # It uses a temp=1 if episodeStep < tempThreshold,
    # and thereafter uses temp=0.
    # Return trainExamples: 
    #     a list of examples of the form (canonicalBoard,pi,v)
    #     pi is the MCTS informed policy vector.
    #     v is +1 if the player eventually won the game, else -1.
    #--------------------------------------------------------------
    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold) # temperature
            
            # Get Action Probability
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            
            # Expand Symmetry States
            sym = self.game.getSymmetries(canonicalBoard, pi)
            
            # Append Board State and Action Probability in Example
            for b,p in sym:
                trainExamples.append([b, p, self.curPlayer, None])
                
            # Decide Action
            action = np.random.choice(len(pi), p=pi) # (np.arange(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            
            # Check Game
            r = self.game.getGameEnded(board, self.curPlayer) #result
                # case) self.curPlayer ==  1; Black
                #     r= 1 if Black won
                #     r=-1 if Black lost or draw
                #     r= 0 if not ended
                # case) self.curPlayer == -1; White
                #     r= 1 if White won
                #     r=-1 if White lost or draw
                #     r= 0 if not ended

            if r!=0:
               #return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]
               #return [(x[0],x[1],r*((-1)**(x[2]!=self.curPlayer))) for x in trainExamples]
                ret = [(x[0],x[1],r*((-1)**(x[2]!=self.curPlayer))) for x in trainExamples]
                return ret #(board state), (probability), (result)
                # result:  which replaces x[2]
                # curP x[2] r  result    
                #   1   1   1    1      Black won 
                #   1   1  -1   -1      Black lost or draw
                #   1  -1   1   -1      Black won          (canonically reversed)
                #   1  -1  -1    1      Black lost or draw (canonically reversed)
                #  -1   1   1   -1      White won 
                #  -1   1  -1    1      White lost or draw
                #  -1  -1   1    1      White won          (canonically reversed)
                #  -1  -1  -1   -1      White lost or draw (canonically reversed)
                
    #----------------------------------------------------------------
    # numIters iterations with numEps episodes of self-play
    #----------------------------------------------------------------
    # Retrains neural network with examples in trainExamples
    # (which has a maximium length of maxlenofQueue).
    # Pits the new neural network against the old one and accepts it
    # only if it wins >= updateThreshold fraction of games.
    #----------------------------------------------------------------
    def learn(self, iterCount):
        # Load Train Examples if last iteration exists
        if iterCount >= 1:
            iterStr = str(iterCount)
            fileTrain = os.path.join(args.file_folder, iterStr + args.file_train)
            if os.path.isfile(fileTrain):
                print("Load Previous Train File: %s" % fileTrain)
                self.loadTrainExamples(fileTrain)
            else:
                print("Can't find Previous Train File: %s" % fileTrain)
                sys.exit()            

        # Repeat Multiple Iterations
        while iterCount < self.args.numIters:
            iterCount += 1 # Initial iterCount is zero
            
            # Print Iteration Counter
            print('------ITER ' + str(iterCount) + '------')
            
            # Examples of the iteration
            if not self.skipFirstSelfPlay or iterCount > 1:
                # Train Example Container
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                # For multiple Episodes
                for eps in range(self.args.numEps):
                    # Print Episode Counter
                    print("\rSelf Play Eps = %d/%d" % (eps+1, self.args.numEps), end="")
                    # Reset Search Tree
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    # Execute One Episode
                    iterationTrainExamples += self.executeEpisode()
                print("")
                # Store the Iteration Examples to the History 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            # Remove the oldest trainExamples if length of trainExamplesHistory overs limit.
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)

            # Backup History to a File
            self.saveTrainExamples(iterCount)
            
            # Shuffle Examples before Training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e) # append from tail
            shuffle(trainExamples)

            # Training New Network, keeping a copy of the old one
            self.nnet.save_checkpoint(self.args.file_folder, self.args.file_model_temp)
            self.pnet.load_checkpoint(self.args.file_folder, self.args.file_model_temp)
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('MATCHING AGAINST PREVIOUS VERSION')
            match = Match(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = match.playGames(self.args.matchCompare)

            print('NEW/PREV WINS : %d/%d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(self.args.file_folder, self.args.file_model_temp) #Revert
            else:
                print('ACCEPTING NEW MODEL')
                fileModel = str(iterCount) + self.args.file_model
                self.nnet.save_checkpoint(self.args.file_folder, fileModel)
                self.nnet.save_checkpoint(self.args.file_folder, self.args.file_model_best)                

            # Update Iteration File
            self.handleIterationFile(iterCount)

    #-------------------------------
    # Save Train Example File
    #-------------------------------
    def saveTrainExamples(self, iteration):
        iterStr = str(iteration)
        fileTrain = os.path.join(self.args.file_folder, iterStr + self.args.file_train)
        # Save Python Object
        with open(fileTrain, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self, filePath):
        # Load Python Object
        with open(filePath, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
        # examples based on the model were already collected (loaded)
        self.skipFirstSelfPlay = True
        
    #----------------------------
    # Handle Iteration File
    #----------------------------
    def handleIterationFile(self, iterCount_W):
        fileIter = os.path.join(self.args.file_folder, self.args.file_iter)
        # First Check if iterCount_W is zero
        if iterCount_W == 0:
            if os.path.isfile(fileIter):
                with open(fileIter, "r") as f:
                    iterCount_R = int(f.read())
                print("Found Previous Iteration: Last Iter = %d" % iterCount_R)                    
            else:
                with open(fileIter, "w") as f:
                    f.write(str(iterCount_W))
                print("Create New Iteration File")                                    
                iterCount_R = 0
        # Update Iteration File if iterCount_W is not zero
        else:
            with open(fileIter, "w") as f:
                f.write(str(iterCount_W))
            print("Update Iteration File: Iter = %d" % iterCount_W)
            iterCount_R = iterCount_W
        # Return
        return iterCount_R

#==========================
# Main Routine
#==========================
if __name__=="__main__":
    # Initialize
    board_size = args.boardSize
    g = ReversiGame(board_size)
    nnet = nn(g)
    teach = Teach(g, nnet, args)
    
    # Check Model Directory
    if os.path.isdir(args.file_folder):
        print("Found Model Directory")
    else:
        print("Not Found Model Directory, Create %s" % args.file_folder)
        os.mkdir(args.file_folder)
        
    # Check Iteration File
    iterCount = teach.handleIterationFile(0)

    # Do Learning
    teach.learn(iterCount)

#===========================================
# End of Program
#===========================================
