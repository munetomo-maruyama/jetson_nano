#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general

import math
import numpy as np
import pdb # Use 'pdb.set_trace()' to break
EPS = 1e-8

#==========================================
# Class: MCTS, Monte Carlo Tree Search
#==========================================
class MCTS():
    #--------------------
    # Initialization
    #--------------------
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
            # stores Q values for s,a (expected reward for taking the action a)
        self.Nsa = {}
            # stores #times edge s,a was visited
        self.Ns = {}
            # stores #times board s was visited
        self.Ps = {}
            # stores initial policy (returned by neural net)
        self.Es = {}
            # stores game.getGameEnded ended for board s
        self.Vs = {}
            # stores game.getValidMoves for board s

    #----------------------------------------------------
    # Performs numMCTSSims simulations of MCTS
    # starting from canonicalBoard.
    #   Returns:
    #       probs: a policy vector where 
    #              the probability of the ith action is
    #              proportional to Nsa[(s,a)]**(1./temp)
    #----------------------------------------------------
    def getActionProb(self, canonicalBoard, temp=1):
        # Do MCTS iterations for numMCTSSims times
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)
            
        # Get Visited Count for the canonicalBoard
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        # If temperature is zero, only a maximum counts is taken
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs # Return Probability list
            
        # Else, make almost uniform distribution
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs # Return Probability list

    #-----------------------------------------------------------------
    # One iteration of MCTS:
    #-----------------------------------------------------------------
    # It is recursively called till a leaf node is found. 
    # The action chosen at each node is one that has the
    # maximum Upper Confidence Bound(UCB) 'u' for each (s,a).
    # Once a leaf node is found, the neural network is called
    # to return an initial policy P and a value v for the state.
    # This value is propogated up the search path.
    # In case the leaf node is a terminal state, the outcome is
    # propogated up the search path.
    # The values of Ns, Nsa, Qsa are updated.
    # NOTE: the return values are the negative of the value of
    #   the current state. This is done since v is in [-1,1]
    #   and if v is the value of a state for the current player,
    #   then its value is -v for the other player.
    #   During the recursive call, player takes each turn alternately.
    # Returns:
    #    v: the negative of the value of the current canonicalBoard
    #-----------------------------------------------------------------
    def search(self, canonicalBoard):
        # Get Game State from Canonical Board
        s = self.game.stringRepresentation(canonicalBoard)
        
        # Store status of Game Ended
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            
        # If Game Ended? Terminate Node
        if self.Es[s]!=0:
            return -self.Es[s] # Negative value for next player
            
        # If Leaf Node?
        if s not in self.Ps:
            # Get Policy and Value for the Canonical Board from NN
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            # Masking Invalid Moves
            self.Ps[s] = self.Ps[s] * valids
            
            # Re-Normalize Policy
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
                
            # If all valid moves were masked
            # make all valid moves equally probable
            else:
                # All valid moves may be masked 
                # if either the NNet architecture is insufficient
                # or overfitting or something else.
                # If you have got dozens or hundreds of these messages
                # you should pay attention to NNet and training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
                
            # Store Value and Visit Count in Leaf Node
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            # Return Negative Value for Next Player
            return -v

        # If not Leaf Node
        valids = self.Vs[s]
        cur_best = -float('inf') # Maxium Negative
        best_act = -1
        
        # Pick the Action with the highest Upper Confidence Bound(UCB)
        # (UCB = Expected Reward + Bonus for selected Action)
        for a in range(self.game.getActionSize()):
            if valids[a]: # If the Action is Valid one
                # Calculate UCB
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
                else: # If Q is Zero
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                # Update Best
                if u > cur_best:
                    cur_best = u
                    best_act = a
                    
        # Move according to best Action, and Change Player
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        
        # Recursive Call this tree search
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s)

        # Update Node Data
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
            self.Nsa[(s,a)] += 1
        else: # Leaf
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
            
        # Accumulate Visit Count
        self.Ns[s] += 1
        
        # Return Negative Value for Next Player
        return -v

#===========================================
# End of Program
#===========================================
