from TMSAnalysis.Clustering import SignalArray
import numpy as np
import copy

debug = False

class Clustering:

    def __init__(self, sig_array):
        self.ct = 3.0 #Time difference for which to consider this the same event
        self.sig_array = sig_array
        self.bundles  = {'X':[], 'Y':[]}
        self.clusters = []

    def GetNumClusters(self):
        return len(self.clusters)

    def Is3DEvent(self):
        for cluster in self.clusters:
            if cluster.GetPos1D('X')<-900 or cluster.GetPos1D('Y')<-900:
                return False
        return True

    def GetNumber3D(self):
        count=0
        for cluster in self.clusters:
            if cluster.GetPos1D('X')>-900 and cluster.GetPos1D('Y')>-900:
                count+=1
        return count

    def Cluster(self):

        #First check if clustering is neccesary
        if self.sig_array.GetNX()==0 and self.sig_array.GetNY()==0:
            #no signals = no clusters
            return
        
        if debug: 
            print("==================================================")
            self.sig_array.Print()
        #First create bundles of like channels
        #This is based on proximity and time
        self.CreateBundles("X")    
        self.CreateBundles("Y")
    
        #Now combine the X/Y bundles to form clusters 
        self.CreateClusters()

        if debug: 
            print("==================================================")

        if debug:
            if self.Is3DEvent(): input("===============>>>>>Pause")
        return

    def CreateBundles(self, name):
        curr_array = copy.copy(self.sig_array.GetSigArray(name)) 
        for sig in curr_array:
            if (self.bundles[name])==0:
                sig_array = SignalArray.SignalArray()
                sig_array.AddSignal(sig)
                self.bundles[name].append(sig_array)
            else:
                isBundled = False
                for i,bundle in enumerate(self.bundles[name]):
                    if bundle.CheckNeighbors(sig, self.ct):
                        self.bundles[name][i].AddSignal(sig)
                        isBundled = True
                        break
                if not isBundled:
                    sig_array = SignalArray.SignalArray()
                    sig_array.AddSignal(sig)
                    self.bundles[name].append(sig_array)
        
        #Check for bundles which can be combinined
        self.RefineBundles(name)


    def RefineBundles(self,name):
        #Check for bundles which can be combinined
        #this can happen if the signals in a bundle are not ordered sequentially
        for i,ibundle in enumerate(self.bundles[name]):
            for j,jbundle in enumerate(self.bundles[name]):
                if i==j: continue
                for sig in jbundle.GetSigArray(name):
                    if ibundle.CheckNeighbors(sig, self.ct):
                        ibundle.AddBundle(jbundle)
                        self.bundles[name].remove(jbundle)
                        break

    def CheckMults(self):
        for sig in self.sig_array.GetSigArray("XY"):
            count = 0
            for cluster in self.clusters:
                if cluster.Contains(sig):
                    count+=1
            #sig.energy *= 1.0/count
            for ic,cluster in enumerate(self.clusters):
                cluster.DivideSigs(sig, count)

    def CreateClusters(self):
        
        for ix, xbundle in enumerate(self.bundles['X']):
            cluster = copy.deepcopy(xbundle)
            ctime = cluster.GetTime()
            
            for iy, ybundle in enumerate(self.bundles['Y']):
                ytime = ybundle.GetTime()
                if abs(ctime-ytime)<self.ct:
                    cluster.AddBundle(copy.deepcopy(ybundle))
                    ybundle.isClustered=True
                    break
            
            self.clusters.append(cluster)
        
        #Now loop and make a cluster for each unmatched ybundle
        for iy, ybundle in enumerate(self.bundles['Y']):
            if ybundle.isClustered: continue
            cluster = copy.copy(ybundle)
            self.clusters.append(cluster)
        
        #Cross checks for signals added to more than 1 cluster
        #The energy of the signal is than dividied by the number of occurances
        self.CheckMults()

        if debug:
            for ci,cluster in enumerate(self.clusters):
                print("")
                print("Cluster-%i"% (ci))
                cluster.Print()
                print("")

        
