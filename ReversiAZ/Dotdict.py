#===========================================
# ReversiAZ : Reversi Game by Reinforcement 
#-------------------------------------------
# Rev.0.1 2019.09.26 Munetomo Maruyama
#-------------------------------------------
# Copyright (C) 2018 Surag Nair
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# Based on https://github.com/suragnair/alpha-zero-general

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

#===========================================
# End of Program
#===========================================
