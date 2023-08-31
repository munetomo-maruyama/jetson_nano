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

import tkinter as tk
import numpy as np
import sys
from threading import Thread, Event
from time import sleep
from enum import Enum
from Board import Board
from Game import ReversiGame
from NNet import NNetWrapper as NNet
from MCTS import MCTS
from Dotdict import *
import pdb # Use 'pdb.set_trace()' to break

#--------------------------------
# Define Arguments as Dictionary
#--------------------------------
"""
args = dotdict({
    'boardSize' : 6,
    'model_dir' : './model',
    'model_file': 'best_model.tar',
})

"""
"""
args = dotdict({
    'boardSize' : 6,
    'model_dir' : './model_trained',
    'model_file': '6x6_153checkpoints_best.pth.tar',
})
"""

args = dotdict({
    'boardSize' : 8,
    'model_dir' : './model_trained',
    'model_file': '8x8_100checkpoints_best.pth.tar',
})

#-----------------------
# Enumeration
#-----------------------
class GameState(Enum):
    INIT = 0
    SELECT_BLACK = 1
    SELECT_WHITE = 2

#=======================
# Class: Play as GUI
#=======================
class PlayGUI(object):
    #----------------
    # Initialization
    #----------------
    def __init__(self, master, board_size):
        # Initialize Internal State
        self.quit = False
        self.board_size = board_size
        self.clicked = False
        self.gamestate = GameState.INIT

        # Initialize Game
        self.game = ReversiGame(self.board_size)
        self.board = self.game.getInitBoard()
        self.nnet = NNet(self.game)

        # Load Model
        self.nnet.load_checkpoint(args.model_dir, args.model_file)
        self.args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.nnet_player = lambda x: np.argmax(
            self.mcts.getActionProb(x, temp=0)
        )
                
        # Generate Canvas
        self.cvs_board = tk.Canvas(
            width  = self.board_size * 80, 
            height = self.board_size * 80, 
            highlightthickness = 0
        )
        self.cvs_board.bind("<Button-1>", self.on_clicked)
        self.cvs_board.grid(
            row = 0, column = 0,
            columnspan = 4,
            padx = 0, pady = 0
        )
        self.draw_board()
        
        # Generate Message Label
        self.mes_str = tk.StringVar()
        self.lbl_mes = tk.Label(textvariable = self.mes_str)
        self.lbl_mes.grid(
            row = 1, column = 0,
            columnspan = 4, sticky = tk.W + tk.E,
            padx = 5, pady = 5
        )
        self.mes_str.set(u'Preparing...')
        
        # Generate Quit Button
        self.btn_start_black = tk.Button(
            text='Quit', state = tk.NORMAL,
            command = self.on_button_quit)
        self.btn_start_black.grid(
            row = 2, column = 0,
            sticky = tk.W + tk.E,
            padx = 5, pady = 5
        )
        
        # Generate Start as Black (first) Button
        self.btn_start_black = tk.Button(
            text='Start as BLACK', state = tk.DISABLED,
            command = self.on_button_start_as_black)
        self.btn_start_black.grid(
            row = 2, column = 1,
            sticky = tk.W + tk.E,
            padx = 5, pady = 5
        )
        
        # Generate Start as White (last) Button
        self.btn_start_white = tk.Button(
            text='Start as WHITE', state = tk.DISABLED,
            command = self.on_button_start_as_white)
        self.btn_start_white.grid(
            row = 2, column = 2,
            sticky = tk.W + tk.E,
            padx = 5, pady = 5
        )
        
        # Generate Pass Button
        self.btn_pass = tk.Button(
            text='Pass', state = tk.DISABLED,
            command = self.on_button_pass)
        self.btn_pass.grid(
            row = 2, column = 3,
            sticky = tk.W + tk.E,
            padx = 5, pady = 5
        )
        
        # Start Main Thread Loop       
        self.callback(self.MainThreadLoop)

    #----------------------
    # Thread Support
    #----------------------
    def callback(self, tgt):
        self.th = Thread(target = tgt)
        self.ev = Event()
        self.th.start()

    #----------------------
    # Close Window
    #----------------------
    def on_closing(self):
        root.destroy()
        self.quit = True
        if (hasattr(self, 'th')):
            self.ev.set() # Resume Thread
            self.th.join()
        sys.exit()
        
    #----------------------
    # Draw Board
    #----------------------
    def draw_board(self):
        n = self.board_size
        self.cvs_board.delete('all')
        self.cvs_board.create_rectangle(
            0, 0, n*80, n*80, 
            width = 0.0, fill = '#00B200'
        )
        for i in range(1, n + 1):
            self.cvs_board.create_line(
                0, i*80, n*80, i*80, 
                width = 1.0, fill = '#000000'
            )
            self.cvs_board.create_line(
                i*80, 0, i*80, n*80, 
                width = 1.0, fill = '#000000'
            )
        if (hasattr(self, 'board')):
            for y in range(n):
                for x in range(n):
                    piece = self.board[y][x]
                    posx = x * 80 + 10
                    posy = y * 80 + 10
                    if piece == 1: # BLACK
                        self.cvs_board.create_oval(
                            posx, posy, posx + 60, posy + 60,
                            width = 1.0,
                            outline = '#000000', 
                            fill = '#000000'
                        )
                    if piece == -1: # WHITE
                        self.cvs_board.create_oval(
                            posx, posy, posx + 60, posy + 60,
                            width = 1.0,
                            outline = '#000000', 
                            fill = '#FFFFFF'
                        )
    #----------------------
    # Print Board History
    #----------------------
    def print_board(self):
        n = self.board_size
        for i in range(2*n-1): print("-", end="")
        print("")
        for y in range(n):
            for x in range(n):
                piece = self.board[y][x]
                if piece ==  1: print("B ",end="")
                elif piece == -1: print("W ",end="")
                else: print("- ",end="")
            print("")
        for i in range(2*n-1): print("-", end="")
        print("")        
    
    #-----------------------
    # On Push Button Quit
    #-----------------------
    def on_button_quit(self):
        self.on_closing()
        
    #-----------------------------------------
    # On Push Button Start as Black (first)
    #-----------------------------------------
    def on_button_start_as_black(self):
        self.gamestate = GameState.SELECT_BLACK
        self.btn_start_black.configure(state = tk.DISABLED)
        self.btn_start_white.configure(state = tk.DISABLED)
        # Resume Thread
        self.ev.set()
            
    #-----------------------------------------
    # On Push Button Start as White (last)
    #-----------------------------------------
    def on_button_start_as_white(self):
        self.gamestate = GameState.SELECT_WHITE
        self.btn_start_black.configure(state = tk.DISABLED)
        self.btn_start_white.configure(state = tk.DISABLED)
        # Resume Thread
        self.ev.set()

    #-----------------------------------------
    # On Push Button Pass
    #-----------------------------------------
    def on_button_pass(self):
        self.do_pass = True
  
    #-----------------------------------------
    # Main Thread Loop
    #-----------------------------------------
    def MainThreadLoop(self):
        self.mes_str.set(u'Select BLACK(1st) or WHITE(2nd).')
        self.btn_start_black.configure(state = tk.NORMAL)
        self.btn_start_white.configure(state = tk.NORMAL)
        
        # Main Loop
        while True:
            # Wait for Event
            self.ev.wait()
            self.ev.clear()
            if (self.quit == True):
                return
            # Select Player
            if self.gamestate == GameState.SELECT_BLACK:
                self.player1 = self.HumanPlayer
                self.player2 = self.nnet_player
            else:
                self.player1 = self.nnet_player
                self.player2 = self.HumanPlayer
            # Playing
            curPlayer = 1
            self.board = self.game.getInitBoard()
            self.draw_board()
            self.print_board() # print history     
            sleep(0.5)
            while self.game.getGameEnded(self.board, curPlayer) == 0:
                if (curPlayer == 1):
                    self.mes_str.set(u'BLACK\'s Turn.')
                    action = self.player1(self.game.getCanonicalForm(self.board, curPlayer))
                else:
                    self.mes_str.set(u'WHITE\'s Turn.')
                    action = self.player2(self.game.getCanonicalForm(self.board, curPlayer))
                if (self.quit == True):
                    return
                self.board, curPlayer = self.game.getNextState(self.board, curPlayer, action)
                self.draw_board()
                self.print_board() # print history     
            # Game Over
            if (self.game.getScore(self.board, 1) > 0):
                self.mes_str.set(u'BLACK won. Play again?')
            elif (self.game.getScore(self.board, -1) > 0):
                self.mes_str.set(u'WHITE won. Play again?')
            else:
                self.mes_str.set(u'Draw. Play again?')
            # Enable Start Button
            self.btn_start_black.configure(state = tk.NORMAL)
            self.btn_start_white.configure(state = tk.NORMAL)
        
    #---------------------------------------
    # On Clicked in Board by Human Player
    #---------------------------------------
    def on_clicked(self, event):
        self.clicked = True
        self.event = event   

    #---------------------------------------
    # Human Player
    #---------------------------------------
    def HumanPlayer(self, board):
        valid = self.game.getValidMoves(board, 1)
        # Not Pass
        if (valid[self.game.n ** 2] == 0):
            # Show Possible Position
            for i in range(len(valid)):
                if valid[i]:
                    x = int(i % self.game.n)
                    y = int(i / self.game.n)
                    posx = x * 80 + 35
                    posy = y * 80 + 35
                    self.cvs_board.create_oval(
                        posx, posy, posx + 10, posy + 10,
                        width = 1.0,
                        outline = '#000000', 
                        fill = '#000000'
                    )
            # Wait for Mouse Click
            self.clicked = False
            while True:
                if (self.quit == True):
                    a = self.game.n ** 2
                    break
                if (self.clicked == True):
                    x = int(self.event.x / 80)
                    y = int(self.event.y / 80)
                    a = self.game.n * y + x 
                    if (valid[a]):
                         break
        # Pass
        else:
            self.do_pass = False
            self.btn_pass.configure(state = tk.NORMAL)
            self.mes_str.set(u'You should Pass.')
            while True:
                if (self.quit == True):
                    a = self.game.n ** 2
                    break
                if (self.do_pass == True):
                    self.do_pass = False
                    self.btn_pass.configure(state = tk.DISABLED)
                    a = self.game.n ** 2
                    break
        # Return Positon
        return a

#==========================
# Main Routine
#==========================
if __name__=="__main__":
    board_size = args.boardSize
    root = tk.Tk()
    root.title(u"Reversi Zero")
    app = PlayGUI(root, board_size) # do not use pack() which has grid inside
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
        
#===========================================
# End of Program
#===========================================
