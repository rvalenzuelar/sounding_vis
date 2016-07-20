#!/home/raul/miniconda3/envs/py27/bin/python -W ignore::VisibleDeprecationWarning

"""
Created on Wed Jul 20 11:22:43 2016

@author: raul
"""
import numpy as np
import sounding as so
import Meteoframes as mf
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['legend.handletextpad'] = 0.2
rcParams['mathtext.default'] = 'sf'

scale=1.2
fig,ax = plt.subplots(figsize=(5*scale,5*scale))

infiles3,_ = so.get_sounding_files('3', homedir='/localdata')
infiles7,_ = so.get_sounding_files('7', homedir='/localdata')


first = True
for f in infiles3:
    df = mf.parse_sounding2(f)
    x = np.expand_dims(df.bvf_moist.values,axis=1)*10000
    y = np.expand_dims(df.index.values,axis=1)
    ax.plot(x,y,color=(1,0,0,0.5),lw=1)
    top = 2000 # [m]
    top_idx = np.where(y == top)[0]
    if first is True:    
        prof = x[:top_idx]
        first = False
    else:
        prof = np.hstack((prof,x[:top_idx]))
        
meanx = np.expand_dims(np.nanmean(prof,axis=1),axis=1)
y2 = y[:top_idx]
ax.plot(meanx,y2,color=(1,0,0,1),lw=3,label='23-24Jan01(n=7)')

first = True    
for f in infiles7:
    df = mf.parse_sounding2(f)
    x = np.expand_dims(df.bvf_moist.values,axis=1)*10000
    y = np.expand_dims(df.index.values,axis=1)
    ax.plot(x,y,color=(0,0.8,0,0.5),lw=1)
    top = 2000 # [m]
    top_idx = np.where(y == top)[0]
    if first is True:    
        prof = x[:top_idx]
        print prof.shape
        first = False
    else:
        prof = np.hstack((prof,x[:top_idx]))
 
meanx = np.expand_dims(np.nanmean(prof,axis=1),axis=1)
y2 = y[:top_idx]
ax.plot(meanx,y2,color=(0,0.8,0,1),lw=3,label='17Feb01(n=11)')       
        
ax.set_xlim([-4,4])
ax.set_ylim([0,2000])

ax.axvline(0,color='k',linestyle=':',lw=3)
ax.set_ylabel('Altitude MSL [m]')
ax.set_xlabel(r'$N^2 [x10^{-4} s^{-2}]$')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

#plt.show()

fname='/home/raul/Desktop/N2.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')
