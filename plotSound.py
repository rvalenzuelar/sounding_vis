"""
	Plot NOAA balloon soundings. Files
	have extension tsv

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import itertools
import metpy.plots as metplt
import matplotlib.pyplot as plt
import numpy as np
import os 
import datetime as dt


''' set directory and input files '''
base_directory='/Users/raulv/Desktop/SOUNDING'
print base_directory
usr_case = raw_input('\nIndicate case number (i.e. 1): ')
case='case'+usr_case.zfill(2)
casedir=base_directory+'/'+case
out=os.listdir(casedir)
# out.sort()
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
	sounding = pd.read_table(file_sound,skiprows=36,header=None)
	sounding.drop(19 , axis=1, inplace=True)	
	sounding.columns=col_names

	''' assing metadata, but does not propagate out of def '''
	units = dict(zip(col_names, col_units))
	for n in col_names:
		sounding[n].units=units[n]

	# ''' add thermodynamics '''
	# theta = thermo.theta(C=meteo.temp,hPa=meteo.press)
	# thetaeq = thermo.theta_equiv(C=meteo.temp,hPa=meteo.press)
	# meteo.loc[:,'theta'] = pd.Series(theta,index=meteo.index)	
	# meteo.loc[:,'thetaeq'] = pd.Series(thetaeq,index=meteo.index)	

	return sounding

def plot_skew(sounding,date):

	s=sounding
	hgt=sounding.Height # [m]
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


for f in file_sound:
	print f
	foo = parse_dataframe(f)
	fs = os.path.basename(f)
	raw_date=fs[:-4]
	date = dt.datetime.strptime(raw_date, "%Y%m%d_%H%M%S")
	# date = ' '
	plot_skew(foo,date)

plt.show()






