"""
	Plot NOAA balloon soundings as a 
	time-height section. 
	Files have extension tsv.

	Raul Valenzuela
	September, 2015
"""


import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import os 
from datetime import datetime, timedelta
import sys

import Thermodyn as thermo
import Meteoframes as mf

from scipy.interpolate import interp1d

from  scipy.interpolate import Rbf


# import seaborn as sns

# ''' set color codes in seaborn '''
# sns.set_color_codes()
# rc = {'axes.titlesize': 24,
# 		'axes.labelsize': 18,
# 		'ytick.labelsize':16,
# 		'xtick.labelsize':16}
# sns.set(rc=rc)



def main():
	
	file_sound = get_sounding_files(None)

	''' raw soundings '''
	soundarray,_,_,_,_ = get_raw_array('thetaeq', file_sound)	
	fig,ax=plt.subplots()
	ax.imshow(soundarray,interpolation='none',origin='top',aspect='auto')
	ax.invert_xaxis()
	plt.subplots_adjust(left=0.125,right=0.9)	
	plt.draw()	

	''' interpolated '''
	soundarray2,_,_ = get_interp_array('thetaeq',files=file_sound)	

	''' image plot '''
	# fig,ax=plt.subplots()
	# ax.imshow(soundarray2,interpolation='none',origin='top',aspect='auto')
	# ax.invert_xaxis()
	# plt.subplots_adjust(left=0.125,right=0.9)	
	# plt.draw()	

	''' contourf plot '''
	rows,cols = soundarray2.shape
	X,Y = np.meshgrid(range(cols),range(rows))
	fig,ax=plt.subplots()
	ax.contourf(X,Y,soundarray2)
	ax.invert_xaxis()
	plt.draw()	


	plt.show(block=False)


def get_sounding_files(usr_case):

	''' set directory and input files '''
	# base_directory='/Users/raulv/Desktop/SOUNDING'
	base_directory='/home/rvalenzuela/BALLOON'
	if not usr_case:
		print base_directory
		usr_case = raw_input('\nIndicate case number (i.e. 1): ')
	case='case'+usr_case.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	file_sound=[]
	for f in out:
		if f[-3:]=='tsv': 
			file_sound.append(casedir+'/'+f)	

	return file_sound

def get_raw_array(soundvar,file_sound):

	''' height grid '''
	hgtgrid = np.asarray(range(12, 4000,12))

	''' time grid '''
	dates=np.arange('2001-01-23 16:00', '2001-01-24 04:40', dtype='datetime64[20m]')

	var=[] # 2D list
	timestamps=[]
	top_limit=4100 #[m]
	file_sound.sort()
	for f in file_sound:
		
		''' timestamps '''
		fname = os.path.basename(f)
		raw_date=fname[:-4].split('_')[-2:]
		y=raw_date[0][:4]
		m=raw_date[0][4:6]
		d=raw_date[0][6:8]
		hh=int(raw_date[1][0:2])
		mm=int(raw_date[1][2:4])
		nearest = round_to_nearest(mm,20)
		if nearest == 60:
			hh += 1
			mm = 0
		else:
			mm = nearest
		raw_date=y+'-'+m+'-'+d+'T'+str(hh).zfill(2)+':'+str(mm).zfill(2)+':'+str(0).zfill(2)
		timestamps.append(np.datetime64(raw_date))

		''' variable '''
		df = mf.parse_sounding(f)
		sv= df[soundvar][ df.index<top_limit ].values
		Z=df.index[ df.index<top_limit ].values

		''' interpolate to a common vertical grid '''
		f=interp1d(Z,sv)
		svinterp=f(hgtgrid)

		''' add to list '''
		var.append(svinterp)

	''' get index of each sounding date '''
	tidx = []
	for t in timestamps:
		tidx.append(np.where(dates == t)[0][0])

	''' assign soundings to 2D array '''
	soundarray = np.full((hgtgrid.size,dates.size),np.nan)
	for n,i in enumerate(tidx):
		soundarray[:,i] = var[n]

	return soundarray,var,tidx,hgtgrid,dates


def get_interp_array(soundvar,**kwargs):

	for key,value in kwargs.iteritems():
		if key == 'case':
			file_sound=get_sounding_files(value)
		elif key == 'files':
			file_sound = value

	soundarray,var,tidx,hgtgrid,dates = get_raw_array(soundvar,file_sound)

	''' create column variables to feed Rbf '''
	x=np.repeat(tidx,len(hgtgrid),axis=0)
	x = x[:,np.newaxis]

	y=hgtgrid[:,np.newaxis].T
	y= np.tile(y,len(tidx))
	y= np.reshape(y,(y.size,1))
	
	z = np.asarray(var)
	z = np.reshape(z,(z.size,1))

	''' when soundings have nans '''
	nans = np.isnan(z)
	nonnan = np.where(~nans)[0]

	''' 2D interpolation '''	
	rbf = Rbf(x[nonnan],y[nonnan],z[nonnan])
	soundarray2 = np.copy(soundarray)
	for d in range(len(dates)):
		if d not in tidx:
			xi=np.repeat(d,len(hgtgrid))
			zi = rbf(xi,hgtgrid)
			soundarray2[:,d] = zi	

	if np.any(nans):
		soundarray2 = fill_nan_gaps(soundarray2)

	return soundarray2,hgtgrid,dates

def fill_nan_gaps(array):
	
	''' fills nans by averaging closest neighbors'''

	gx,gy=np.where(np.isnan(array))
	for i,j in zip(gx,gy):
		precolumn= array[i,j-1]
		poscolumn = array[i,j+1]
		new_value = (precolumn+poscolumn)/2
		array[i,j] = new_value
	return array

def round_to_nearest(num, base):
	''' see tzaman answer in stackoverflow '''
	n = num + (base//2)
	return n - (n % base)


def get_dates(start,end,delta):

	dates=[]
	if delta == timedelta(hours=0):
		dates.append(start)
		return dates
	else:
		foo=start
		dates.append(start)
		while foo<=end:
			foo += delta
			dates.append( foo)
		return dates

''' start '''
if __name__ == "__main__":
	main()