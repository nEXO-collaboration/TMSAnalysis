import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.widgets import Slider, Button, RadioButtons


#load a viewer using an infile that is an 
#hdf5 file containing a waveform dataframe, processed
#from binary by the CryoAsicFile class

class CryoAsicEventViewer:
	def __init__(self, infile):
		self.infile = infile
		if(self.infile[-2:] != "h5" and self.infile[-4:] != "hdf5"):
			print("Input file to CryoAsicEventViewer is not an hdf5 file: " + self.infile)
			return

		print("loading hdf5 file " + self.infile)
		self.df = pd.read_hdf(self.infile, key='raw')
		print("Done loading")

		self.nevents_total = len(self.df.index)

		self.sf = 1 #MHz "sf": sampling_frequency
		self.dT = 1.0/self.sf



	#plots the event in the same way that the CRYO ASIC GUI
	#event viewer plots, with a 2D greyscale hist using the
	#*channel numbers as the y axis bins and the time on x axis. 
	#this is distinct from other plots in that usually the positions
	#of the strips are used as the y axis; in this way, this is more
	#what the ASIC sees, ordering based on the asic channel IDs. 
	def plot_event_rawcryo(self, evno):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]
		chs = ev["Channels"]
		waves = ev["Data"]
		#sort them simultaneously by channel number
		chs, waves = (list(t) for t in zip(*sorted(zip(chs, waves))))
		nch = len(chs)
		nsamp = len(waves[0])
		times = np.arange(0, nsamp*self.dT, self.dT)


		#this is a fancy pandas way of making a 2D hist. I always
		#have trouble finding the meshgrid way of doing this. 
		#please modify this if you have a better way. 
		plot_df = pd.DataFrame()
		for i, ch in enumerate(chs):
			for j, t in enumerate(times):
				e = pd.Series()
				e["t"] = t #time 
				e["ch"] = ch #channel
				e["v"] = waves[i][j] #voltage value at that time
				plot_df = plot_df.append(e, ignore_index=True)
		plot_matrix = plot_df.pivot('ch', 't', 'v')

		fig, ax = plt.subplots(figsize=(12,8))
		heat = ax.imshow(plot_matrix, cmap='viridis',\
			extent=[plot_df['t'].min(), plot_df['t'].max(), plot_df['ch'].min(), plot_df['ch'].max()],\
			aspect=0.5)

		cbar = fig.colorbar(heat, ax=ax)
		cbar.set_label("ADC counts", labelpad=3)
		ax.set_xlabel("time (us)")
		ax.set_ylabel("ASIC ch number")
		ax.set_title("event number " + str(evno))


		#functions for controlling slider bars
		axvmin = plt.axes([0.25, 0.025, 0.65, 0.01], facecolor='green')
		axvmax = plt.axes([0.25, 0.04, 0.65, 0.01], facecolor='green')
		vminbar = Slider(axvmin, 'min adc counts', 0, 4096, valinit=plot_df["v"].min())
		vmaxbar = Slider(axvmax, 'max adc counts', 0, 4096, valinit=plot_df["v"].max())
		def update(val):
			vmin = vminbar.val 
			vmax = vmaxbar.val 
			heat.set_clim(vmin, vmax)
			fig.canvas.draw_idle()
		vminbar.on_changed(update)
		vmaxbar.on_changed(update)
		reset = plt.axes([0.0, 0.025, 0.1, 0.04])
		button = Button(reset, 'Reset', color='green', hovercolor='0.975')
		def reset(event):
			vminbar.reset()
			vmaxbar.reset()
		button.on_clicked(reset)

		plt.show()

	#plots the event with a 2D hist where each tile has two subplots:
	#one for the X strips and one for the Y strips. Time on x axis, 
	#local channel position on Y axis. Finds all tiles and generates
	#subplots for each. 
	def plot_event_xysep(self, evno):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]
		types = ev["ChannelTypes"]
		pos = ev["ChannelPositions"]
		waves = ev["Data"]
		nch = len(pos)
		nsamp = len(waves[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		#this is a fancy pandas way of making a 2D hist. I always
		#have trouble finding the meshgrid way of doing this. 
		#please modify this if you have a better way. 
		xstrip_plot_df = pd.DataFrame()
		ystrip_plot_df = pd.DataFrame()
		for i, p in enumerate(pos):
			typ = types[i]
			for j, t in enumerate(times):
				e = pd.Series()
				e["t"] = t #time 
				e["pos"] = p #position
				e["v"] = waves[i][j] #voltage value at that time
				if(typ == 'x'):
					xstrip_plot_df = xstrip_plot_df.append(e, ignore_index=True)
				else:
					ystrip_plot_df = ystrip_plot_df.append(e, ignore_index=True)

		xplot_matrix = xstrip_plot_df.pivot('t', 'pos', 'v')
		yplot_matrix = ystrip_plot_df.pivot('pos', 't', 'v')
		fig, ax = plt.subplots(figsize=(12,8), nrows = 2)
		xheat = ax[0].imshow(xplot_matrix, cmap='viridis',\
			extent=[xstrip_plot_df['t'].min(), xstrip_plot_df['t'].max(), xstrip_plot_df['ch'].min(), xstrip_plot_df['ch'].max()],\
			aspect=0.5)

		yheat = ax[1].imshow(yplot_matrix, cmap='viridis',\
			extent=[ystrip_plot_df['t'].min(), ystrip_plot_df['t'].max(), ystrip_plot_df['ch'].min(), ystrip_plot_df['ch'].max()],\
			aspect=0.5)

		xcbar = fig.colorbar(xheat, ax=ax[0])
		xcbar.set_label("ADC counts", labelpad=3)
		ax[0].set_xlabel("time (us)")
		ax[0].set_ylabel("strip position")
		ax[0].set_title("event number " + str(evno))

		ycbar = fig.colorbar(xheat, ax=ax[0])
		ycbar.set_label("ADC counts", labelpad=3)
		ax[1].set_ylabel("time (us)")
		ax[1].set_xlabel("strip position")
		ax[1].set_title("event number " + str(evno))

		"""
		#functions for controlling slider bars
		axvmin = plt.axes([0.25, 0.025, 0.65, 0.01], facecolor='green')
		axvmax = plt.axes([0.25, 0.04, 0.65, 0.01], facecolor='green')
		vminbar = Slider(axvmin, 'min adc counts', 0, 4096, valinit=plot_df["v"].min())
		vmaxbar = Slider(axvmax, 'max adc counts', 0, 4096, valinit=plot_df["v"].max())
		def update(val):
			vmin = vminbar.val 
			vmax = vmaxbar.val 
			heat.set_clim(vmin, vmax)
			fig.canvas.draw_idle()
		vminbar.on_changed(update)
		vmaxbar.on_changed(update)
		reset = plt.axes([0.0, 0.025, 0.1, 0.04])
		button = Button(reset, 'Reset', color='green', hovercolor='0.975')
		def reset(event):
			vminbar.reset()
			vmaxbar.reset()
		button.on_clicked(reset)
		"""
		plt.show()


	#This is a fancy plot where we remove the "time" domain of the data,
	#and instead plot interleaved X and Y strips together. The color of
	#the strip represents the maximum value of the waveform within the 
	#waveform buffer. 
	def plot_event_tile_maximum(self, evno):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

	#This is a fancy plot where we remove the "time" domain of the data,
	#and instead plot interleaved X and Y strips together. The color of
	#the strip represents the baseline of the waveform within the 
	#waveform buffer. Baseline is calculated as the mean of the 
	#final few microseconds, "baseline_buffer"
	def plot_event_tile_baseline(self, evno, baseline_buffer = 10):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return


	#this plots the waveforms from x and y all on the same plot, 
	#overlayed, but with traces shifted relative to eachother by 
	#some number of ADC counts. if tileno is not none, it only plots
	#one tile, associated with an integer passed as argument
	def plot_event_waveforms_separated(self, evno):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]

		adc_shift = 0 #number of adc counts to shift traces

		chs = ev["Channels"]
		waves = ev["Data"]
		#sort them simultaneously by channel number
		chs, waves = (list(t) for t in zip(*sorted(zip(chs, waves))))
		nch = len(chs)
		nsamp = len(waves[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		fig, ax = plt.subplots(figsize=(10,8))
		curshift = 0
		for i in range(nch):
			ax.plot(times, waves[i] + curshift, label=str(chs[i]))
			#curshift += adc_shift

		ax.set_xlabel('time (us)')
		ax.set_ylabel("channel shifted adc counts")


		plt.show()

	











