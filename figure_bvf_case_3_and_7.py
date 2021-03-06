#!/home/raul/miniconda3/envs/py27/bin/python -W ignore::VisibleDeprecationWarning

"""
Created on Wed Jul 20 11:22:43 2016

@author: raul
"""
import numpy as np
import sounding as so
import Meteoframes as mf
import matplotlib.pyplot as plt
from rv_utilities import discrete_cmap
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['legend.handletextpad'] = 0.2
rcParams['mathtext.default'] = 'sf'

scale=1.2
fig,axes = plt.subplots(2,1,sharex=True,figsize=(5*scale,10*scale))
axes[0].set_gid('(a) 23-24Jan01')
axes[1].set_gid('(b) 17Feb01')

nobs=('n=7','n=11')

infiles3,_ = so.get_sounding_files('3', homedir='/localdata')
infiles7,_ = so.get_sounding_files('7', homedir='/localdata')

cmap = discrete_cmap(7, base_cmap='Set1')
color=(cmap(0),cmap(1))

infiles=(infiles3,infiles7)


for n,ax in enumerate(axes):

    first = True
    for f in infiles[n]:
        df = mf.parse_sounding2(f)
        x = np.expand_dims(df.bvf_moist.values,axis=1)*10000
        y = np.expand_dims(df.index.values,axis=1)
        ax.plot(x,y,color=color[n],lw=0.5)
        top = 2000 # [m]
        top_idx = np.where(y == top)[0]
        if first is True:    
            prof = x[:top_idx]
            first = False
        else:
            prof = np.hstack((prof,x[:top_idx]))
    meanx = np.expand_dims(np.nanmean(prof,axis=1),axis=1)
    y2 = y[:top_idx]
    ax.plot(meanx,y2,color=color[n],lw=3)
    xpos = 0.08
    ax.text(xpos,0.9,ax.get_gid(),
            fontsize=15,
            weight='bold',
            transform=ax.transAxes,
            )
    ax.text(xpos+0.07,0.85,nobs[n],
            fontsize=15,
            weight='bold',
            transform=ax.transAxes,            
            )     
     
    ax.set_xlim([-4,4])
    ax.set_ylim([0,2000])
    ax.axvline(0,color='k',linestyle=':',lw=3)
    
axes[0].set_ylabel('Altitude MSL [m]')
axes[1].set_xlabel(r'$N_{m}^{2} [x10^{-4} s^{-2}]$')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)

plt.subplots_adjust(hspace=0.05)

#plt.show()

fname='/home/raul/Desktop/fig_N2.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')
