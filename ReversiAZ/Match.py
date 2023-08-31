#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general

import pdb # Use 'pdb.set_trace()' to break

#=======================================
# Class: Match
#     2 players match against each other
#=======================================
class Match():
    #-------------------------------
    # Initialization
    #-------------------------------
    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    #-----------------------------------
    # Play One Episode of a game
    #     Return  1 if Black won.
    #     Return -1 if White won.
    #     Return  0 if draw.
    #-----------------------------------
    def playGame(self):
        curPlayer = 1 #Black=self.player1, first
        board = self.game.getInitBoard()
        while self.game.getGameEnded(board, curPlayer) == 0:
            players =  self.player1 if curPlayer == 1 else self.player2
            action = players(self.game.getCanonicalForm(board, curPlayer))
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        return self.game.getGameEnded(board, 1)

    #-----------------------------------------------------
    # Play num Episodes of game
    #     player1 starts num/2, player2 starts num/2
    #     Return oneWon : number of games won by player1
    #     Return twoWon : number of games won by player2
    #     Return draws  : number of games won by nobody
    #-----------------------------------------------------
    def playGames(self, num, verbose=False):
        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws  = 0
        # player1 first as Black
        # player2 last  as White
        eps = 0
        for _ in range(num):
            gameResult = self.playGame()
            if gameResult == 1:
                oneWon+=1
            elif gameResult == -1:
                twoWon+=1
            else:
                draws+=1
            print("\rPlaying (Player1 First) Eps = %d/%d" % (eps+1, num), end="")
            eps += 1
        print("")

        # Change Player's turn
        self.player1, self.player2 = self.player2, self.player1
        
        # player1 last  as White
        # player2 first as Black
        eps = 0
        for _ in range(num):
            gameResult = self.playGame()
            if gameResult == 1:
                twoWon+=1                
            elif gameResult == -1:
                oneWon+=1
            else:
                draws+=1
            print("\rPlaying (Player2 First) Eps = %d/%d" % (eps+1, num), end="")
            eps += 1
        print("")

        # Return results
        return oneWon, twoWon, draws

#===========================================
# End of Program
#===========================================
