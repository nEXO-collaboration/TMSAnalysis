import copy
import numpy as np

class Signal:
    def __init__( self, energy, time, ch, pos, name):
        self.energy      = energy
        self.time        = time
        self.channel     = ch
        self.pos         = {"X":pos[0], "Y":pos[1]}
        self.ch_name     = name
        
        if "X" in self.ch_name: self.ctype = "X"
        elif "Y" in self.ch_name: self.ctype = "Y"
        else: self.ctype = "Other"

    def Print(self):
        print("Ch: %s, Time: %.2f, Energy: %.2f, Pos: [%.2f, %.2f]" % (self.ch_name, 
                                                                       self.time, 
                                                                       self.energy,
                                                                       self.pos['X'], self.pos['Y']))

