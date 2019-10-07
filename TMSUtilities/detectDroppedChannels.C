//#include "TTree.h"
//#include "TFile.h"
//#include "NGMHit.h"
//#include <cstdio>
//#include <sstream>

//int main() {
{
  printf("Infile name stream...\n");

  int num_channels = 16;
  int num_evts_per_file = 10000;

  stringstream infile_name_stream;
  infile_name_stream << "/farmshare/user_data/blenardo/struck_data/"
              << "tier1_SIS3316Raw_20181004190032_Ambe_CH1NaI1700V_CH0PSD1100V_36-5inchSTD_PSDTrigg150mVTh_PSDX10_125MHz_1-ngm" 
              << ".root";

  stringstream outfile_name_template_stream;
  outfile_name_template_stream << "/farmshare/user_data/blenardo/struck_data/"
            << "tier1_SIS3316Raw_20181004190032_Ambe_CH1NaI1700V_CH0PSD1100V_36-5inchSTD_PSDTrigg150mVTh_PSDX10_125MHz_1-ngm" 
            << "_";

  string infile_name = infile_name_stream.str();
  string outfile_name_template = outfile_name_template_stream.str();

  printf("Opening infile...\n");
  TFile * f = new TFile( infile_name.c_str() );

  printf("Getting tree..\n");
  TTree * hit_tree = (TTree *)f->Get("HitTree");  
  NGMHitv8 * hit = new NGMHitv8();

  printf("Setting HitTree branch address...\n");
  hit_tree->SetBranchAddress("HitTree",&hit);


  printf("Number of entries: %d\n",(int) hit_tree->GetEntries());
  printf("This means there are %d events\n",(int) hit_tree->GetEntries()/num_channels);

  int n_entries = hit_tree->GetEntries();

  int n_entries_to_process = n_entries;
//  int n_entries_to_process = (num_evts_per_file*num_channels)*5;

  int i_entry = 0;
  int i_outfile = 0;
  char outfilename[1000];
  TFile * out_file;
  TTree * out_hit_tree;


  while( i_entry < n_entries_to_process ) {
 
       printf("%d\n",i_entry);         
 
 
       int j_entry_out = 0;
       int channel_map[num_channels];
       int i_current_ch;
       int i_current_ch_plus_sixteen;
//       printf("============================= OUTFILE %d ================================\n",i_outfile);
       //while( j_entry_out < (num_evts_per_file*num_channels) && (i_entry + j_entry_out) < n_entries ) {

       while( j_entry_out < (num_evts_per_file*num_channels) && i_entry < n_entries_to_process ) {

          int event_ch_index = i_entry % num_channels;
          hit_tree->GetEntry(i_entry);
          i_current_ch = hit->GetChannel();
          hit_tree->GetEntry(i_entry+16);
          i_current_ch_plus_sixteen = hit->GetChannel();
          if( i_current_ch != i_current_ch_plus_sixteen ) {
              printf("Dropped channel at entry %d\n",i_entry);
              //return;
          }
          j_entry_out += 1;
          i_entry += 1;
       }
       //i_entry += 1;
       i_outfile += 1;
  }

//return 0;
}
