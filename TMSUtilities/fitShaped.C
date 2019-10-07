#include "TF1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TTree.h"
#include "TFile.h"
#include "TMinuit.h"
#include "TRandom3.h"
#include "TStopwatch.h"
#include "TStyle.h"
#include <iostream>
#include <vector>
#include <sstream>
using namespace std;

Double_t fitShaped(Double_t *x, Double_t *par){
 // 0: integral of the output voltage
 // 1: starting time
 // 2: differential time
 // 3: integration time
 // 4: constant term/ noise pedestal  
//double fenzi = par[2]*TMath::Exp((x[0]-par[1])/par[2])-(par[2]+(x[0]-par[1])*(par[2]-par[3])/par[3])*TMath::Exp((x[0]-par[1])/par[3]);
//  double fenmu=(par[2]-par[3])*(par[2]-par[3]);
  double fenzi=TMath::Exp(-(x[0]-par[1])/par[2])-TMath::Exp(-(x[0]-par[1])/par[3]);
  double fenmu=par[2]/(par[2]-par[3]);
  double result=par[0]*fenzi/fenmu+par[4];
  if(x[0]<par[1]) return par[4];
  else return result;
}
