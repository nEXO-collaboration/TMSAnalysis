import pandas as pd
import numpy as np
import pickle

class LightMap:
    def __init__(self, bins=10, tile_size=45.0):
        self.bins = bins
        self.tile_half_size = tile_size/2

    def load_files(self, fchroma, freduced):
        self.chroma_sim = pd.DataFrame.from_dict(pd.read_pickle(fchroma)).set_index('index')
        self.reduced = pd.read_hdf(freduced).iloc[self.chroma_sim.index]

    def get_matrix(self, pos_x, pos_y, pos_z, light_energy):
        self.lim_x = np.linspace(-self.tile_half_size,self.tile_half_size,self.bins)
        self.lim_y = np.linspace(-self.tile_half_size,self.tile_half_size,self.bins)
        self.lim_z = np.linspace(pos_z.min(),pos_z.max(),self.bins+1)
        bin_x = np.digitize(pos_x,self.lim_x)
        bin_y = np.digitize(pos_y,self.lim_y)
        bin_z = np.digitize(pos_z,self.lim_z)
        vxl = np.array((bin_x,bin_y,bin_z)).T
        voxel = np.unique(vxl,axis=0)
        mean_light_energy = np.mean(light_energy)
        self.ly_matrix = np.ones((self.bins+1,self.bins+1,self.bins+2))
        n_light_ev = np.zeros((self.bins+1,self.bins+1,self.bins+2))

        for vl in voxel:
            selection_idx = np.where(np.sum((vxl == vl).reshape(-1,3),axis=1) == 3)[0]
            light_energy_voxel = light_energy.iloc[selection_idx]
            self.ly_matrix[vl[0],vl[1],vl[2]] = np.mean(light_energy_voxel)/mean_light_energy
            #self.ly_matrix[vl[0],vl[1],vl[2]] = np.mean(light_energy_voxel)
            #n_light_ev[vl[0],vl[1],vl[2]] = selection_idx.shape[0]
        #self.ly_matrix /= np.mean(self.ly_matrix[1:,1:,1:-1])
        #np.save('sim/light_map_uniform',ly_matrix)

    def rescale_light_energy(self, pos_x, pos_y, pos_z, light_energy):
        bin_x = np.digitize(pos_x,self.lim_x)
        bin_y = np.digitize(pos_y,self.lim_y)
        bin_z = np.digitize(pos_z,self.lim_z)
        vxl = np.array((bin_x,bin_y,bin_z)).T
        for i,vl in enumerate(vxl):
            light_energy.iloc[i] /= self.ly_matrix[vl[0],vl[1],vl[2]]
        return light_energy

    def get_xy_position_from_light(self, sipm_center, light_array):
        missing_pos = np.zeros((len(light_array),2))
        j = 0
        for _,ev in light_array.iterrows():
            ev_loc = np.zeros((len(ev.keys()),2))
            for i,k in enumerate(ev.keys()):
                if len(sipm_center[k])>3:
                    sipm_center[k] = np.mean(sipm_center[k],axis=0)
                ev_loc[i] = sipm_center[k][:2]*ev[k]
            missing_pos[j] = np.sum(ev_loc,axis=0)/sum(ev)
            j += 1
        return missing_pos

    def apply_photon_threshold(self,thsd):
        return self.chroma_sim.where(self.chroma_sim>thsd,0)
