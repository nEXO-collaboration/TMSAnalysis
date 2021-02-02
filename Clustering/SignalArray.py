from StanfordTPCAnalysis.Clustering import Signal
import numpy as np
import copy

class SignalArray:
    def __init__(self):
        #self.xsigs = []
        #self.ysigs = []
        self.sigs = {'X':[], 'Y':[]}
        self.isClustered = False

    def GetEnergy(self, ctype='XY'):
        energy = 0
        for sig in self.GetSigArray(ctype):
            energy+=sig.energy
        return energy

    def GetTime(self):
        energy = self.GetEnergy()
        time = 0
        for sig in self.GetSigArray('XY'):
            time += sig.time*sig.energy
        if energy>0: return time/energy
        else: return 0.0

    def GetTimeRMS(self):
        event_time = self.GetTime()
        time_sq    = 0
        energy     = self.GetEnergy()
        for sig in self.GetSigArray('XY'):
            time_sq += ((sig.time-event_time)*sig.energy)**2
        if energy>0:return np.sqrt(time_sq)/energy
        else: return 0.0

    def GetPosRMS(self,ctype):
        energy     = self.GetEnergy()
        event_pos  = self.GetPos1D(ctype)
        pos_sq     = 0
        if energy>0:
            for sig in self.GetSigArray(ctype):
                if sig.pos[ctype]>-900:
                    pos_sq += ((sig.pos[ctype]-event_pos)*sig.energy)**2.0
            return np.sqrt(pos_sq)/energy
        else:
            return 0.0

    def GetPos1D(self,ctype):
        pos = 0
        energy = self.GetEnergy(ctype)
        if energy>0:
            for sig in self.GetSigArray(ctype):
                pos += sig.pos[ctype]*sig.energy
            return pos/energy
        else:
            return -999.0

    def GetPos2D(self):
        pos = {"X":self.GetPos1D("X"), "Y":self.GetPos1D("Y")}
        return pos


    def AddXSignal(self, sig): 
        self.sigs['X'].append(sig)
    def AddYSignal(self,sig):  
        self.sigs['Y'].append(sig)
    
    def AddSignal(self, sig):
        if "Y" in sig.ch_name: 
            self.AddYSignal(sig)
        elif "X" in sig.ch_name: 
            self.AddXSignal(sig)
        else: 
            print("Untracked channel name %s ?????"%sig.ch_name)
    
    def AddBundle(self, bundle):
        for sig in bundle.GetSigArray("XY"):
            self.AddSignal(sig)

    def GetNX(self): 
        return len(self.sigs['X'])
    def GetNY(self): 
        return len(self.sigs['Y'])

    def GetSigArray(self,name):
        if name=='X': return self.sigs['X']
        elif name=='Y': return self.sigs['Y']
        else: return (self.sigs['X']+self.sigs['Y'])

    def Print(self):
        for sig in self.GetSigArray('XY'):
            sig.Print()
        print("Pos:[%.2f, %.2f], Time:%.2f, Energy:%.2f"%(self.GetPos1D('X'), self.GetPos1D('Y'), 
                                                          self.GetTime(), 
                                                          self.GetEnergy()))

    def Contains(self, sig):
        #Check if signal is already in the list
        for nsig in self.GetSigArray(sig.ctype):
            if sig.ch_name == nsig.ch_name:
                if abs(sig.time-nsig.time)<0.01 and abs(sig.energy-nsig.energy)<0.01:
                    return True
        return False

    def CheckNeighbors(self, sig, ct):
        #Check to see if the neighboring channels have a signal in the array
        neighbor = []
        for nsig in self.GetSigArray(sig.ctype):
            if abs(self.GetTime()-sig.time) > ct:
                continue
            if abs(nsig.channel-sig.channel)==1:
                return True
        return False

    def DivideSigs(self, sig, counts):
        #print("Do divide")
        for index,nsig in enumerate(self.GetSigArray(sig.ctype)):
            if sig.ch_name == nsig.ch_name:
                if abs(sig.time-nsig.time)<0.01 and abs(sig.energy-nsig.energy)<0.01:
                    self.sigs[sig.ctype][index].energy *= 1.0/counts
                    #print("%s %.2f"%(sig.ch_name,self.sigs[sig.ctype][index].energy))
                    #if counts>1: input("counts %i"%counts)




