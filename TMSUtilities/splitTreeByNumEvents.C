//#include "TTree.h"
//#include "TFile.h"
//#include "NGMHit.h"
//#include <cstdio>
//#include <sstream>

//int main() {
{
  printf("Infile name stream...\n");

  int num_channels = 16;
  int num_evts_per_file = 1000;

  stringstream infile_name_stream;
  infile_name_stream << "/farmshare/user_data/blenardo/struck_data/"
              << "tier1_SIS3316Raw_20181004190032_Ambe_CH1NaI1700V_CH0PSD1100V_36-5inchSTD_PSDTrigg150mVTh_PSDX10_125MHz_1-ngm_0015" 
              << ".root";

  stringstream outfile_name_template_stream;
  outfile_name_template_stream << "/farmshare/user_data/blenardo/struck_data/"
            << "tier1_SIS3316Raw_20181004190032_Ambe_CH1NaI1700V_CH0PSD1100V_36-5inchSTD_PSDTrigg150mVTh_PSDX10_125MHz_1-ngm_0015" 
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
//  int n_entries_to_process = (num_evts_per_file*num_channels)*2;

  int i_entry = 0;
  int i_outfile = 0;
  char outfilename[1000];
  TFile * out_file;
  TTree * out_hit_tree;

  // Global loop. "While there is still stuff in the input file..."
  while( i_entry < n_entries_to_process ) {
 
       printf("%d\n",i_entry);         
 
       sprintf(outfilename,"%s%04d.root",outfile_name_template.c_str(),i_outfile);
       out_file = new TFile(outfilename,"RECREATE");
       out_hit_tree = new TTree();

       out_hit_tree->Branch("HitTree",&hit);
       out_hit_tree->SetName("HitTree");     
 
       int j_entry_out = 0;
       printf("============================= OUTFILE %d ================================\n",i_outfile);
       printf("%s\n",outfilename); 
       //while( j_entry_out < (num_evts_per_file*num_channels) && (i_entry + j_entry_out) < n_entries ) {
       int evt_start_ch = 0;
       int i_current_ch, i_current_ch_pnch, i_current_ch_mnch;

       // File loop. "While the output file is not yet filled up and we haven't yet exhausted the input..."
       while( j_entry_out < (num_evts_per_file*num_channels) && i_entry < n_entries_to_process ) {
            if( j_entry_out % 1000 == 0 ) printf("Processed %d events.\n",j_entry_out);
            hit_tree->GetEntry(i_entry);
            i_current_ch = hit->GetChannel();

            // Get the channel number of the entry num_channels ahead of the current event. 
            if( i_entry + num_channels < n_entries_to_process ) {
                hit_tree->GetEntry(i_entry+num_channels);
                i_current_ch_pnch = hit->GetChannel();
            } else {
                // if we're at the end of the dataset, just skip the last bit of stuff.
                i_entry += 1;
                continue;
            }

            // If the channel numbers match (no dropped channels), perform the save normally.
            if( i_current_ch == i_current_ch_pnch ) {
                hit_tree->GetEntry(i_entry);
                out_hit_tree->Fill();
                j_entry_out += 1;
                i_entry += 1;
            } else {
                printf("Error detected! Entry %d.\n",i_entry+num_channels);
                // If the channel numbers don't match,
                //      - finish current event
                //      - get timestamp of next event
                //      - Skip all entries with this timestamp
                hit_tree->GetEntry(i_entry);
                int good_evt_timestamp = (int) hit->GetRawClock();
                int current_evt_timestamp = good_evt_timestamp;
                while( current_evt_timestamp == good_evt_timestamp ){
                   out_hit_tree->Fill();
                   j_entry_out += 1;
                   i_entry += 1;
                   hit_tree->GetEntry(i_entry);
                   current_evt_timestamp = (int) hit->GetRawClock();
                }
                int bad_evt_timestamp = current_evt_timestamp;
                while( current_evt_timestamp == bad_evt_timestamp ){
                   i_entry += 1;
                   hit_tree->GetEntry(i_entry);
                   current_evt_timestamp = (int) hit->GetRawClock();
                }               
            }
                              
       }

       out_file->cd();
       out_hit_tree->Write();
       out_file->Close();
       delete out_hit_tree;
       delete out_file;
       //i_entry += 1;
       i_outfile += 1;
  }

//return 0;
}
