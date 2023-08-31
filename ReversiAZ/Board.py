#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2008 Eric P. Nichols (board logic)
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general
# Based on Board Logic orignal by Eric P. Nichols, Feb 8, 2008.

import pdb # Use 'pdb.set_trace()' to break

#===================================
# Class: Board (Reversi Logic)
#===================================
# Board data:
#  1=black, -1=white, 0=empty
#  first dim is row , 2nd is column:
#     pieces[7][1] is the square in column 2,
#     at the opposite end of the board in row 8.
# Squares are stored and manipulated as (x,y) tuples.
# x is the column, y is the row.
class Board():
    #----------------------------------------------------------
    # list of all 8 directions on the board, as (x,y) offsets
    #----------------------------------------------------------
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]
    
    #-----------------------------------------------------
    # Initialization : Set up initial board configuration
    #-----------------------------------------------------
    def __init__(self, n):
        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n
        # Set up the initial 4 pieces.
        self.pieces[int(self.n/2)  ][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)  ] = 1
        self.pieces[int(self.n/2)  ][int(self.n/2)  ] = -1;
        self.pieces[int(self.n/2)-1][int(self.n/2)-1] = -1;

    #----------------------------------------
    # add [][] indexer syntax to the Board
    #----------------------------------------
    def __getitem__(self, index): 
        return self.pieces[index]

    #-------------------------------------------------
    # Counts the # pieces of the given color
    # (1 for black, -1 for white, 0 for empty spaces)
    #-------------------------------------------------
    def countDiff(self, color):
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[y][x]==color:
                    count += 1
                if self[y][x]==-color:
                    count -= 1
        return count

    #-------------------------------------------------
    # Returns all the legal moves for the given color.
    #-------------------------------------------------
    def get_legal_moves(self, color):
        # Storage for the legal moves.
        moves = set() # Collective Data Type
        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[y][x]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    #----------------------------------------------------------
    # If there are legal moves in current board for the color ?
    #----------------------------------------------------------
    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[y][x]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    #------------------------------------------------------------------
    # Returns all the legal moves that use the given square as a base.
    # If the given square is (3,4) and it contains a black piece,
    # and (3,5) and (3,6) contain white pieces, and (3,7) is empty,
    # one of the returned moves is (3,7) because everything from there
    # to (3,4) is flipped.
    #------------------------------------------------------------------
    def get_moves_for_square(self, square):
        (x,y) = square
        # Determine the color of the piece.
        color = self[y][x]
        # Skip empty source squares.
        if color==0:
            return None
        # Search all possible directions.
        moves = []
        for direction in self.__directions:
            flips = []
            (x,  y ) = square
            (dx, dy) = direction
            move = None
            while True:
                x = x + dx
                y = y + dy
                if (x < 0 or x >= self.n or y < 0 or y >= self.n):
                    break
                if self[y][x] == 0:
                    if flips:
                         move = (x, y)
                    break
                elif self[y][x] ==  color:
                    break
                elif self[y][x] == -color:
                    flips.append((x, y))
            if move:
                moves.append(move)
        # Return the generated move list
        return moves

    #----------------------------------------------------------------
    # Perform the given move on the board; flips pieces as necessary.
    # color gives the Color of the piece to play (1=black,-1=white)
    #----------------------------------------------------------------
    def execute_move(self, move, color):
        # Start at the new piece's square and follow it
        # on all 8 directions to look for a piece allowing flipping.
        flips = [move]
        for direction in self.__directions:
            (x,  y ) = move
            (dx, dy) = direction
            flip = []
            while True:
                x = x + dx
                y = y + dy
                if (x < 0 or x >= self.n or y < 0 or y >= self.n):
                    break
                if self[y][x] == 0:
                    break
                if self[y][x] == -color:
                    flip.append((x, y))
                elif self[y][x] == color:
                    flips.extend(flip) # append only element
                    break
        for x, y in flips:
            self[y][x] = color

#===========================================
# End of Program
#===========================================
