#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general

from __future__ import print_function
import sys
import numpy as np
from Board import Board
import pdb # Use 'pdb.set_trace()' to break

#========================
# Class Reversi Game
#========================
class ReversiGame():
    #--------------------
    # Initialization
    #--------------------
    def __init__(self, n):
        self.n = n

    #------------------------------------
    # Returns initial board (numpy board)
    #------------------------------------
    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    #---------------------
    # Returns Board Size
    #---------------------
    def getBoardSize(self):
        return (self.n, self.n) # tuple type

    #--------------------------
    # Returns Number of Actions
    #--------------------------
    def getActionSize(self):
        return self.n*self.n + 1 # +1 means "pass"

    #--------------------------------------------------------------
    # Returns next (board, color) if player took action on board.
    # Note: action must be a valid one.
    #--------------------------------------------------------------
    def getNextState(self, board, color, action):
        if action == self.n*self.n: # pass
            return (board, -color)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (action % self.n, int(action / self.n))
        b.execute_move(move, color)
        return (b.pieces, -color)

    #----------------------------------------
    # Returns all valid moves for the color
    #----------------------------------------
    def getValidMoves(self, board, color):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(color)
        if len(legalMoves) == 0: # pass
            valids[-1] = 1 # set last element
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * y + x] = 1
        return np.array(valids)

    #--------------------------------------
    # If the game has ended?
    #   Returns  0 : not ended
    #   Returns  1 : color won
    #   Returns -1 : color lost or draw
    #--------------------------------------
    def getGameEnded(self, board, color):
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(color):
            return 0
        if b.has_legal_moves(-color):
            return 0
        if b.countDiff(color) > 0:
            return 1 # if player won
        return -1    # if player lost or draw

    #-----------------------------------------------
    # Returns canonical formed board for the player
    #-----------------------------------------------
    def getCanonicalForm(self, board, player):
        # if player==1 return board, else if player==-1 return -board
        return player * board

    #---------------------------------------------------
    # Returns 8 symmetry boards with action probability
    #---------------------------------------------------
    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # +1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n)) # remove last element
        l = []
        # Make 8 Symmetries
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])] # add last element
        return l

    #---------------------------------------------
    # Convert numpy canonical board to one string
    #---------------------------------------------
    def stringRepresentation(self, board):
        return board.tostring()

    #------------------------------
    # Get Score from Board State
    #------------------------------
    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

#===========================================
# End of Program
#===========================================
