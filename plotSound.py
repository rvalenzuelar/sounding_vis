"""
	Plot NOAA balloon soundings. Files
	have extension tsv

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import itertools
import metpy.plots as metplt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import os 
import datetime as dt
import Thermodyn as thermo


''' set color codes in seaborn '''
sns.set_color_codes()

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

def get_var_names(file_sound):

	names=[]
	with open(file_sound,'r') as f:
		for line in itertools.islice(f, 15, 34):
			foo = line.split()
			if foo[0]=='T':
				'''pandas has a T property so
				needs to be replaced'''
				names.append('TE')
			else:
				names.append(foo[0])
	return names

def get_var_units(file_sound):

	units=[]
	with open(file_sound,'r') as f:
		for line in itertools.islice(f, 15, 34):
			foo = line.split()
			units.append(foo[1])
	return units

def parse_dataframe(file_sound):

	col_names=get_var_names(file_sound)
	col_units=get_var_units(file_sound)

	''' read tabular file '''
	raw_sounding = pd.read_table(file_sound,skiprows=36,header=None)
	raw_sounding.drop(19 , axis=1, inplace=True)	
	raw_sounding.columns=col_names
	hgt = raw_sounding['Height']
	sounding=raw_sounding[['TE','TD','RH','u','v','P','MR']]
	sounding.index=hgt

	''' replace nan values '''
	nan_value = -32768.00
	sounding = sounding.applymap(lambda x: np.nan if x == nan_value else x)

	''' add potential temperature '''
	theta = thermo.theta(K=sounding.TE, hPa=sounding.P)
	sounding.loc[:,'theta'] = pd.Series(theta, index=sounding.index)	
	thetaeq = thermo.theta_equiv(K=sounding.TE, hPa=sounding.P)		
	sounding.loc[:,'thetaeq'] = pd.Series(thetaeq,index=sounding.index)

	''' add Brunt-Vaisala frequency '''
	bvf_dry= thermo.bv_freq_dry(theta=sounding.theta, agl_m=hgt, depth_m=100,centered=True)
	bvf_moist= thermo.bv_freq_moist(K=sounding.TE, hPa=sounding.P, 
										agl_m=hgt, depth_m=100,centered=True)
	sounding = pd.merge(sounding,bvf_dry,left_index=True,right_index=True,how='inner')

	return sounding

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
	skew.plot(pres, TD, 'g')
	skew.plot_barbs(pres, U2, V2,y_clip_radius=0.005)
	skew.plot_dry_adiabats()
	skew.plot_moist_adiabats()
	skew.ax.set_ylim(1000, 500)
	skew.ax.set_xlim(-10, 30)
	skew.ax.set_ylabel('Pressure [hPa]')
	skew.ax.set_xlabel('Temperature [degC]')

	l1='Balloon sounding at XXX'
	l2='\nDate: ' + date.strftime('%Y %m %d %H:%M:%S UTC')
	plt.suptitle(l1+l2)

	plt.draw()


def plot_thermo(sounding,date):

	hgt=sounding.index # [m]
	pres=sounding.P #[hPa]
	TE=sounding.TE - 273.15 # [C]
	TD=sounding.TD - 273.15 # [C]
	theta=sounding.theta
	thetaeq=sounding.thetaeq
	U=sounding.u.values
	V=sounding.v.values
	BVFd=sounding.bvf_dry.values

	fig,ax = plt.subplots(1,5,sharey=True,figsize=(8.5,11))

	ax[0].plot(TE,hgt,label='Temp')
	ax[0].plot(TD,hgt,label='Dewp')
	# plt.legend()
	ax[0].set_xlim([-30,20])
	ax[0].set_ylim([0,5000])
	add_minor_grid(ax[0])

	ax[1].plot(theta,hgt)
	ax[1].set_xlim([270,330])
	ax[1].set_ylim([0,5000])
	add_minor_grid(ax[1])

	ax[2].plot(thetaeq,hgt)
	ax[2].set_xlim([270,330])
	ax[2].set_ylim([0,5000])
	add_minor_grid(ax[2])

	ax[3].plot(U,hgt,label='u')
	ax[3].plot(V,hgt,label='v')
	# plt.legend()
	ax[3].set_xlim([-10,40])
	ax[3].set_ylim([0,5000])
	add_minor_grid(ax[3])

	ax[4].plot(BVFd*10000,hgt,'o')
	# plt.legend()
	ax[4].set_xlim([-6,6])
	ax[4].set_ylim([0,5000])
	add_minor_grid(ax[4])

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


for f in file_sound:
	print f
	foo = parse_dataframe(f)
	fs = os.path.basename(f)
	raw_date=fs[:-4]
	date = dt.datetime.strptime(raw_date, "%Y%m%d_%H%M%S")
	# plot_skew(foo,date)
	plot_thermo(foo,date)
	# break

plt.show()






