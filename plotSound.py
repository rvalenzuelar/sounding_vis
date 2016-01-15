"""
	Plot NOAA balloon soundings. 
	Files have extension tsv

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import metpy.plots as metplt
import metpy.calc as metcal

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import os 
import datetime as dt
import sys

import Thermodyn as thermo
import Meteoframes as mf

from scipy.interpolate import interp1d
from metpy.units import units, concatenate

''' set color codes in seaborn '''
sns.set_color_codes()

rc = {'axes.titlesize': 24,
		'axes.labelsize': 18,
		'ytick.labelsize':13,
		'xtick.labelsize':13}

# mpl.rcParams.update(rc)

sns.set(rc=rc)

''' set directory and input files '''
# base_directory='/Users/raulv/Desktop/SOUNDING'
base_directory='/home/rvalenzuela/BALLOON'
print base_directory
usr_case = raw_input('\nIndicate case number (i.e. 1): ')
case='case'+usr_case.zfill(2)
casedir=base_directory+'/'+case
out=os.listdir(casedir)
file_sound=[]
for f in out:
	if f[-3:]=='tsv': 
		file_sound.append(casedir+'/'+f)
file_sound.sort()

def main():

	for f in file_sound:
		print f
		df = mf.parse_sounding2(f)

		fname = os.path.basename(f)
		''' removes file extension and split date '''
		raw_date=fname[:-4].split('_')
		''' some files have preffix, so I take only datetime'''
		raw_date = raw_date[-2:]
		if len(raw_date[1]) == 6:
			raw_date=raw_date[0]+raw_date[1]
			date = dt.datetime.strptime(raw_date, "%Y%m%d%H%M%S")
		else:
			raw_date=raw_date[0]+raw_date[1]
			date = dt.datetime.strptime(raw_date, "%Y%m%d%H%M")
		
		# plot_skew(df,date)
		plot_thermo(df,date,top=5.)
		# compare_potential_temp(df,date)
		# break
		# print df
	plt.show()
	# plt.show(block=False)


def compare_potential_temp(sounding,date):

	theta1 = thermo.theta1(K=sounding.TE, hPa=sounding.P)
	theta2 = thermo.theta2(K=sounding.TE, hPa=sounding.P,mixing_ratio=sounding.MR/1000)
		
	thetaeq1 = thermo.theta_equiv1(K=sounding.TE, hPa=sounding.P)		
	thetaeq2 = thermo.theta_equiv2(K=sounding.TE, hPa=sounding.P,
										relh=sounding.RH,mixing_ratio=sounding.MR/1000)	

	foo=pd.DataFrame({'theta1':theta1, 'thetaeq1':thetaeq1,'theta2':theta2, 'thetaeq2':thetaeq2})
	y= foo.index.values

	fig,ax=plt.subplots(figsize=(8.5,11))
	ln1=ax.plot(theta1,y,label='theta=f(temp,press) - W&H')
	ln2=ax.plot(theta2,y,label='theta=f(temp,press, w) - Bolton 1980')
	ln3=ax.plot(thetaeq1,y,label='thetaeq=f(temp,press,ws)  - W&H')
	ln4=ax.plot(thetaeq2,y,label='thetaeq=f(temp,press, w, rh) - Bolton 1980')
	ax.set_xlim([280,320])
	ax.set_ylim([0,5000])
	ax.set_xlabel('Temperature [K]')
	ax.set_ylabel('Altitude [m]')

	ax2=ax.twiny()
	ln5=ax2.plot(sounding.MR,y,sns.xkcd_rgb["amber"], label='mixing ratio')
	ax2.set_xlim([0,8])
	ax2.set_ylim([0,5000])
	ax2.grid(False)
	ax2.set_xlabel('Mixing ratio [g kg-1]')	

	ax3=ax.twiny()
	ln6=ax3.plot(sounding.RH,y,sns.xkcd_rgb["plum purple"], label='relative humidity')
	ax3.set_xlim([0,100])
	ax3.set_ylim([0,5000])
	ax3.grid(False)
	ax3.set_xlabel('Relative humidity [%]')
	ax3.xaxis.set_label_coords(0.5, 1.07)
	ax3.tick_params(direction='out', pad=35)
	lns = ln1+ln2+ln3+ln4+ln5+ln6
	labs = [l.get_label() for l in lns]
	ax3.legend(lns, labs, loc=0)

	plt.subplots_adjust(bottom=0.05,top=0.89)
	datestr=date.strftime('%Y%m%d_%H%M%S')
	plt.title('Comparison of Potential Temperature BBY sounding '+datestr,y=0.99)
	plt.draw()

def plot_skew(sounding,date):

	hgt=sounding.index # [m]
	pres=sounding.P #[hPa]
	TE=sounding.TE - 273.15 # [C]
	TD=sounding.TD - 273.15 # [C]
	U=sounding.u.values
	V=sounding.v.values

	freq=20	
	U2=np.empty(len(U))
	U2[:]=np.NAN
	U2[::freq]=U[::freq]

	V2=np.empty(len(V))
	V2[:]=np.NAN
	V2[::freq]=V[::freq]
	
	fig = plt.figure(figsize=(9, 9))
	skew = metplt.SkewT(fig)

	skew.plot(pres, TE, 'r')
	skew.plot(pres, TD, 'g',linestyle='--')
	skew.plot_barbs(pres, U2, V2)
	p0=np.asarray([1000,900,800,700,600])*units.hPa
	t0=np.asarray([0,5,10,15,20,25]) * units.degC
	skew.plot_dry_adiabats(t0=t0,p=p0,linestyle='-',linewidth=0.8)
	skew.plot_moist_adiabats(t0=t0, p=p0,linestyle='-',linewidth=0.8)
	skew.ax.set_ylim(1014, 600)
	skew.ax.set_xlim(-5, 15)
	skew.ax.set_ylabel('Pressure [hPa]')
	skew.ax.set_xlabel('Temperature [degC]')


	l1='Balloon sounding at Bodega Bay'
	l2='\nDate: ' + date.strftime('%Y-%b-%d %H:%M:%S UTC')
	plt.title(l1+l2)

	plt.draw()

def plot_thermo(sounding,date,**kwarg):

	hgt=sounding.index/1000. # [m]
	pres=sounding.P #[hPa]
	TE=sounding.TE - 273.15 # [C]
	TD=sounding.TD - 273.15 # [C]
	relh=sounding.RH
	mixr=sounding.MR
	theta=sounding.theta
	thetaeq=sounding.thetaeq
	U=sounding.u.values
	V=sounding.v.values
	BVFd=sounding.bvf_dry
	BVFm=sounding.bvf_moist

	fig,ax = plt.subplots(1,5,sharey=True,figsize=(13,8.5))

	hgt_lim=kwarg['top']

	n=0
	ax[n].plot(TE,hgt,label='Temp')
	ax[n].plot(TD,hgt,label='Dewp')
	ax[n].legend()
	ax[n].set_xlim([-30,20])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	ax[n].set_xlabel('Temperature [C]')
	ax[n].set_ylabel('Altitude [km]')

	n=1
	ln1=ax[n].plot(mixr,hgt,label='mixr')
	ax[n].set_xlim([0,8])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	for label in ax[n].xaxis.get_ticklabels()[::2]:
		label.set_visible(False)
	ax[n].set_xlabel('Mixing Ratio [g kg-1]')
	axt=ax[n].twiny()
	ln2=axt.plot(relh,hgt,'g',label='relh')
	axt.set_xlim([0,100])
	axt.set_ylim([0,hgt_lim])	
	axt.set_xlabel('Relative humidity [%]')
	axt.xaxis.set_label_coords(0.5, 1.04)
	axt.grid(False)
	lns = ln1+ln2
	labs = [l.get_label() for l in lns]
	axt.legend(lns, labs, loc=0)

	n=2
	ax[n].plot(theta,hgt,label='Theta')
	ax[n].plot(thetaeq,hgt,label='ThetaEq')	
	ax[n].legend()
	ax[n].set_xlim([280,320])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	for label in ax[n].xaxis.get_ticklabels()[::2]:
		label.set_visible(False)
	ax[n].set_xlabel('Theta [K]')

	n=3
	ax[n].plot(U,hgt,label='u')
	ax[n].plot(V,hgt,label='v')
	ax[n].axvline(x=0,linestyle=':',color='r')
	ax[n].legend()
	ax[n].set_xlim([-10,40])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	ax[n].set_xlabel('Wind Speed [ms-1]')

	n=4
	bvfd=BVFd*10000.
	bvfm=BVFm*10000.
	ax[n].plot(bvfd,hgt,label='dry')
	ax[n].plot(bvfm,hgt,label='moist')	
	ax[n].axvline(x=0,linestyle=':',color='r')
	ax[n].legend(loc=2)
	ax[n].set_xlim([-6,6])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	ax[n].set_xlabel('BVF (x10^-4) [s-2]')

	# f=interp1d(hgt, range(len(hgt)))
	# idx=int(np.ceil(f(1.)))
	# dfslice=bvfm.iloc[0:idx+1]
	# mean = np.nanmean(dfslice)
	# median = np.nanmedian(dfslice)
	# minn = np.amin(dfslice)
	# maxx = np.amax(dfslice)
	# strout='N^2 moist (e-4) lowest 1000 mts: mean={:1.1f}, median={:1.1f}, min={:1.1f}, max={:1.1f}'	
	# print strout.format(mean,median,minn,maxx)	
	# dfslice=bvfd.iloc[0:idx+1]
	# mean = np.nanmean(dfslice)
	# median = np.nanmedian(dfslice)
	# minn = np.amin(dfslice)
	# maxx = np.amax(dfslice)
	# strout='N^2 dry (e-4) lowest 1000 mts: mean={:1.1f}, median={:1.1f}, min={:1.1f}, max={:1.1f}\n'
	# print strout.format(mean,median,minn,maxx)

	# idx=int(np.ceil(f(0.5)))
	# print sounding.iloc[0:idx+1]

	l1='Profile at BBY from sounding '
	l2=date.strftime('%Y-%m-%d %H:%M:%S UTC')
	plt.suptitle(l1+l2)
	plt.subplots_adjust(bottom=0.08,top=0.89,left=0.06, right=0.98)

	plt.draw()

def add_minor_grid(ax):

	# ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
	ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())	
	ax.grid(b=True, which='major', color='w', linewidth=1.0)
	ax.grid(b=True, which='minor', color='w', linewidth=0.5)

def find_nearest2(array,target):

	""" See stackoverflow answer from Bi Rico """
	''' array must be sorted '''
	idx = array.searchsorted(target)
	idx = np.clip(idx, 1, len(array)-1)
	left = array[idx-1]
	right = array[idx]
	idx -= target - left < right - target
	return idx


''' start '''
main()





