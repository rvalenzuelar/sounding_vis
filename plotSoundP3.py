"""
	Plot NOAA balloon soundings. 
	Files have extension tsv

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import metpy.plots as metplt
import metpy.calc as metcal
from metpy.units import units, concatenate

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import os 
from datetime import datetime
import sys

import Thermodyn as thermo
import Meteoframes as mf

from skewt import SkewT as sk

''' set color codes in seaborn '''
sns.set_color_codes()
rc = {'axes.titlesize': 20,
		'axes.labelsize': 18,
		'ytick.labelsize':16,
		'xtick.labelsize':16}
# mpl.rcParams.update(rc)
sns.set(rc=rc)

def main():

	''' raf files are created using matlab function 
	W:/MATLAB/p3_matlab/convert_to_raf_ascii.m '''
	f='/home/rvalenzuela/P3_stdtape/ascii/010123I.std.ascii.raf'
	ini,end = get_sounding_times()

	m=0
	for i,e in zip(ini,end):

		if m>=0:
			sound=mf.parse_acft_sounding(f, i, e, return_interp=True)
			# print sound
			temp=sound.AIR_TEMP.values #[C]
			dewp=sound.DEW_POINT.values #[C]
			wspd=sound.WIND_SPD.values
			wdir=sound.WIND_DIR.values
			u,v = get_components(wspd,wdir)
			pres=sound.AIR_PRESS.values
			lats=sound.LAT.values
			lons=sound.LON.values
			hgt=sound.index.values
			theta=sound.theta.values
			thetaeq=sound.thetaeq.values
			bvf_dry=sound.bvf_dry.values
			bvf_moist=sound.bvf_moist.values
			location=[np.average(lats),np.average(lons)]

			# plot_skew1(temp=temp, dewp=dewp, u=u, v=v, press=pres, date=[i, e],	loc=location)
			plot_skew2(temp=temp, dewp=dewp, u=u, v=v, press=pres, date=[i, e],	loc=location, hgt=hgt)
			# plot_thermo(temp=temp, dewp=dewp, u=u, v=v, hgt=hgt,theta=theta, thetaeq=thetaeq,
			# 			 bvf_dry=bvf_dry, bvf_moist=bvf_moist,press=pres, date=[i, e], loc=location,top=5000)


		m+=1

	
	# compare_potential_temp(df,date)
	# break

	plt.show()
	# plt.show(block=False)


# def plot_skew1(**kwargs):

# 	pres=kwargs['press'] #[hPa]
# 	TE=kwargs['temp'] # [C]
# 	TD=kwargs['dewp'] # [C]
# 	U=kwargs['u'] # [m s-1]
# 	V=kwargs['v'] # [m s-1]
# 	date=kwargs['date']
# 	loc=kwargs['loc']

# 	if np.amax(TE)>30:
# 		TE=TE-273.15
# 	if np.amax(TD)>30:
# 		TD=TD-273.15		

# 	freq=20	
# 	U2=np.empty(len(U))
# 	U2[:]=np.NAN
# 	U2[::freq]=U[::freq]

# 	V2=np.empty(len(V))
# 	V2[:]=np.NAN
# 	V2[::freq]=V[::freq]
	
# 	fig = plt.figure(figsize=(9, 9))
# 	# fig,ax = plt.figure(figsize=(9, 9))
# 	skew = metplt.SkewT(fig)
# 	skew.plot(pres, TE, 'r')
# 	skew.plot(pres, TD, 'g',linestyle='--')
# 	skew.plot_barbs(pres, U2, V2)
# 	p0=np.asarray([1000,900,800,700,600])*units.hPa
# 	t0=np.asarray(range(-10,25,5)) * units.degC
# 	skew.plot_dry_adiabats(t0=t0,p=p0,linestyle='-',linewidth=0.8)
# 	skew.plot_moist_adiabats(t0=t0, p=p0,linestyle='-',linewidth=0.8)
# 	skew.ax.set_ylim(1014, 600)
# 	skew.ax.set_xlim(-10, 12)
# 	skew.ax.set_ylabel('Pressure [hPa]')
# 	skew.ax.set_xlabel('Temperature [degC]')


# 	l1='Sounding from NOAA P3'
# 	l2='\nIni: ' + date[0].strftime('%Y-%b-%d  %H:%M:%S UTC')
# 	l3='\nEnd: ' + date[1].strftime('%Y-%b-%d  %H:%M:%S UTC')
# 	l4='\nLat: '+'{:.2f}'.format(loc[0])+' Lon: '+'{:.2f}'.format(loc[1])

# 	plt.title(l1+l2+l3+l4)
# 	plt.subplots_adjust(top=0.85)

# 	plt.draw()

def plot_skew2(**kwargs):

	pres=kwargs['press'] #[hPa]
	TE=kwargs['temp'] # [C]
	TD=kwargs['dewp'] # [C]
	U=kwargs['u'] # [m s-1]
	V=kwargs['v'] # [m s-1]
	date=kwargs['date']
	loc=kwargs['loc']
	hgt=kwargs['hgt']


	TD[TD>TE]=TE[TD>TE]

	mydata=dict(zip(('hght','pres','temp','dwpt'),(hgt, pres, TE, TD)))
	S=sk.Sounding(soundingdata=mydata)
	S.plot_skewt()
	plt.draw()



def plot_thermo(**kwargs):

	hgt=kwargs['hgt'] #[hPa]
	pres=kwargs['press'] #[hPa]
	TE=kwargs['temp'] # [C]
	TD=kwargs['dewp'] # [C]
	U=kwargs['u'] # [m s-1]
	V=kwargs['v'] # [m s-1]
	date=kwargs['date']
	loc=kwargs['loc']
	theta=kwargs['theta']
	thetaeq=kwargs['thetaeq']
	BVFd=kwargs['bvf_dry']
	BVFm=kwargs['bvf_moist']
	hgt_lim=kwargs['top']

	relh=thermo.relative_humidity(C=TE,Dewp=TD)
	sat_mixr= thermo.sat_mix_ratio(C=TD, hPa=pres)
	mixr = relh*sat_mixr/100.
	

	fig,ax = plt.subplots(1,5,sharey=True,figsize=(11,8.5))


	n=0
	ax[n].plot(TE,hgt,label='Temp')
	ax[n].plot(TD,hgt,label='Dewp')
	ax[n].legend()
	ax[n].set_xlim([-30,20])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	ax[n].set_xlabel('T [C]')
	ax[n].set_ylabel('Altitude [m]')

	n=1
	ln1=ax[n].plot(mixr*1000.,hgt,label='mixr')
	ax[n].set_xlim([0,8])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	for label in ax[n].xaxis.get_ticklabels()[::2]:
		label.set_visible(False)
	ax[n].set_xlabel('MR [g kg-1]')
	axt=ax[n].twiny()
	ln2=axt.plot(relh,hgt,'g',label='relh')
	axt.set_xlim([0,100])
	axt.set_ylim([0,hgt_lim])	
	axt.set_xlabel('RH [%]')
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
	ax[n].set_xlabel('WS [ms-1]')

	n=4
	ax[n].plot(BVFd*10000.,hgt,label='dry')
	ax[n].plot(BVFm*10000.,hgt,label='moist')	
	ax[n].axvline(x=0,linestyle=':',color='r')
	ax[n].legend(loc=2)
	ax[n].set_xlim([-6,6])
	ax[n].set_ylim([0,hgt_lim])
	add_minor_grid(ax[n])
	ax[n].set_xlabel('BVF (x10^-4) [s-1]')

	l1='Sounding from NOAA P3'
	l2='\nIni: ' + date[0].strftime('%Y-%b-%d  %H:%M:%S UTC')
	l3='\nEnd: ' + date[1].strftime('%Y-%b-%d  %H:%M:%S UTC')
	l4='\nLat: '+'{:.2f}'.format(loc[0])+' Lon: '+'{:.2f}'.format(loc[1])

	plt.suptitle(l1+l2+l3+l4,horizontalalignment='right',x=0.9)
	plt.subplots_adjust(bottom=0.06,top=0.89)

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

def get_components(wspd,wdir):

	u = -wspd*np.sin(wdir*np.pi/180.)
	v = -wspd*np.cos(wdir*np.pi/180.)

	return u,v

def get_sounding_times():

	''' sounding times are specified in the 
	flight level documentation '''
	
	ini = [	datetime(2001,01,23,21,28,20),
			datetime(2001,01,23,22,04,00),
			datetime(2001,01,23,23,00,00),
			datetime(2001,01,23,23,13,30),
			datetime(2001,01,23,23,34,30),
			datetime(2001,01,23,23,44,20),
			datetime(2001,01,23,23,58,00),
			datetime(2001,01,24,00,31,40),
			datetime(2001,01,24,00,46,30),
			datetime(2001,01,24,01,01,40),
			datetime(2001,01,24,01,20,20),
			datetime(2001,01,24,01,36,10),
			datetime(2001,01,24,01,56,20),
			datetime(2001,01,24,02,19,00)]

	end = [	datetime(2001,01,23,21,31,50),
			datetime(2001,01,23,22,06,30),
			datetime(2001,01,23,23,04,00),
			datetime(2001,01,23,23,17,50),
			datetime(2001,01,23,23,38,30),
			datetime(2001,01,23,23,53,30),
			datetime(2001,01,24,00,00,30),
			datetime(2001,01,24,00,33,50),
			datetime(2001,01,24,00,48,50),
			datetime(2001,01,24,01,03,50),
			datetime(2001,01,24,01,26,40),
			datetime(2001,01,24,01,39,00),
			datetime(2001,01,24,01,59,30),
			datetime(2001,01,24,02,24,50)]

	return ini,end

''' start '''
main()





