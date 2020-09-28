#########################################################################################
#
#       This script converts to ROOT files the binary files from the struck seriallu.
#       In order to run this script the NGM software needs to be installed and
#       (I think) the the TMS virtual environment deactivated
#
#                                       Jacopo
#
#                                       Usage:
#                       python toRoot.py output_dir binary_files*1.bin
#
#########################################################################################



# From Jason Newby, July 2016

"""
created from toROOT.py
  notes about inheritance and file paths:

  TTask
    NGMModuleBase/NGMModule
      NGMModuleBase/NGMSystem
        NGMModuleCore/NGMMultiFormatReader 
        NGMModuleCore/NGMMultiFormatReader _reader = NGMSIS3316RawReader (public NGMReaderBase)
  
  NGMModuleCore/NGMReaderBase
    NGMModuleCore/NGMSIS3316RawReader


  NGMModuleBase/NGMModule
    NGMModuleCore/NGMHitIO
      NGMModuleCore/NGMHitOutputFile (in NGMHitIO.{h, cc})

NGMModuleCore/NGMSIS3316RawReader.cc fills out the NGMHit object
FIXME -- change NGMHitv6 _waveform to an array/vector? -- maybe make this v8?

"""

import commands
import ROOT
import sys
import os

def toRoot(fname = None, outfolder = None):

    #If input filename is null assume we want to examine the most recent file
    if(fname == None):
        # example name: SIS3316Raw_20160712204526_1.bin
        output  = commands.getstatusoutput("ls -rt SIS3316Raw*_1.bin | tail -n1")
        print "using most recent file, ", output[1]
        fname = output[1]

    print "--> processing", fname

    basename = os.path.basename(fname)
    basename =  os.path.splitext(basename)[0]
    fin = ROOT.NGMMultiFormatReader()
    fin.SetPassName("LIVE")
    mHitOut = ROOT.NGMHitOutputFile("tier1_"+basename,"HitOut")
    mHitOut.setBasePath("./")
    if outfolder is not None:
        mHitOut.setBasePath(outfolder)
    mHitOut.setBasePathVariable("")
    fin.Add(mHitOut)
    fin.initModules()

    fin.OpenInputFile(fname)
    ts = ROOT.TStopwatch()
    ts.Start()
    fin.StartAcquisition(); 
    # calls NGMSIS3316RawReader::ReadAll(), where GetParent()->push() calls NGMNodule::process()
    ts.Print();

if __name__ == "__main__":
    # if no argument is provided, toRoot will find the most recent file.
    # otherwise, loop over arguments
    outfolder = sys.argv[1]
    if len(sys.argv) < 2:
        toRoot()
    elif len(sys.argv) == 2:
        toRoot(outfolder = outfolder)
    else:
        for filename in sys.argv[2:]:
            toRoot(filename,outfolder)
