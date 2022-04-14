import glob, os, pickle, uproot, awkward, argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('inputdir', type=str, help='path of input files')
parser.add_argument('outputfile', type=str, help='output filename')
args = parser.parse_args()
path_to_tree   = args.inputdir
file_chroma = args.outputfile

flist = sorted(glob.glob(path_to_tree + 'SHORTTPC*.root'))
if not '207Bi' in flist[0]:
    photons = {'index':[],'location':[],'p_count':[],'fname':[]}
else:
    photons = {'index':[],'x_location':[],'y_location':[],'z_location':[],'p_count':[],'fname':[]}
flen = 0
for fname in flist:
    tree_p = uproot.concatenate(fname + ':Event/Sim/SimEvent/SimEvent',\
                            expressions=['fNESTLineageX',\
                                         'fNESTLineageY',\
                                         'fNESTLineageZ',\
                                         'fNESTLineageNOP',\
                                         'fGenX',\
                                         'fGenY',\
                                         'fGenZ',\
                                         'fNESTLineageNTE',\
                                         'fNESTLineageT',\
                                         'fNESTLineageType'])

    if not '207Bi' in fname:
        if 'alpha' in fname:
            nop = np.array(awkward.sum(tree_p['fNESTLineageNOP'],axis=1)/10).astype(int)
            ix = np.where(nop>1e5)[0]
        elif 'bipo' in fname:
            nop = np.array(awkward.sum(awkward.where(tree_p['fNESTLineageType']==6,\
                                                     tree_p['fNESTLineageNOP']/10,\
                                                     tree_p['fNESTLineageNOP']),\
                                       axis=1)).astype(int)
            ix = np.where(tree_p['fNESTLineageType'][:,2] == 8)[0]
        photons['location'].extend(np.array([tree_p['fGenX'][ix],tree_p['fGenY'][ix],tree_p['fGenZ'][ix]]).reshape(3,-1).T)
        photons['p_count'].extend(nop[ix])
    else:
        nop = awkward.to_list(tree_p['fNESTLineageNOP'])
        ix = np.arange(len(nop))
        photons['x_location'].extend(awkward.to_list(tree_p['fNESTLineageX']))
        photons['y_location'].extend(awkward.to_list(tree_p['fNESTLineageY']))
        photons['z_location'].extend(awkward.to_list(tree_p['fNESTLineageZ']))
        photons['p_count'].extend(nop)
    photons['fname'].extend(np.repeat(fname,len(ix)))
    photons['index'].extend(ix+flen)

    flen += len(nop)

with open(file_chroma,'wb') as f:
    pickle.dump(photons,f,protocol=2)
