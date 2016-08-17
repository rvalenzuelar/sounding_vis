# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:12:40 2016

@author: raul
"""

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
fig,axes = plt.subplots(4,2,sharex=True,sharey=True,
                        figsize=(8.5*scale,11*scale))
axes = axes.flatten()


axes[0].set_gid('(a) 12-14Jan03 (n={})')
axes[1].set_gid('(b) 21-23Jan03 (n={})')
axes[2].set_gid('(c) 15-16Feb03 (n={})')
axes[3].set_gid('(d) 09Jan04 (n={})')
axes[4].set_gid('(e) 02Feb04 (n={})')
axes[5].set_gid('(f) 16-18Feb04 (n={})')
axes[6].set_gid('(g) 25Feb04 (n={})')
fig.delaxes(axes[-1])

infiles08,_ = so.get_sounding_files('8', homedir='/localdata')
infiles09,_ = so.get_sounding_files('9', homedir='/localdata')
infiles10,_ = so.get_sounding_files('10', homedir='/localdata')
infiles11,_ = so.get_sounding_files('11', homedir='/localdata')
infiles12,_ = so.get_sounding_files('12', homedir='/localdata')
infiles13,_ = so.get_sounding_files('13', homedir='/localdata')
infiles14,_ = so.get_sounding_files('14', homedir='/localdata')


cmap = discrete_cmap(7, base_cmap='Set1')
color = cmap(0)

infiles=(
         infiles08, infiles09,
         infiles10, infiles11,
         infiles12, infiles13,
         infiles14
         )


for ax,infile in zip(axes,infiles):

    first = True
    n = 0
    for f in infile:
        
        try:
            df = mf.parse_sounding2(f)
            x = np.expand_dims(df.bvf_moist.values,axis=1)*10000
            y = np.expand_dims(df.index.values,axis=1)
            ax.plot(x,y,color=color,lw=0.5)
            top = 2000 # [m]
            top_idx = np.where(y == top)[0]
            if first is True:    
                prof = x[:top_idx]
                first = False
                rows,_ = prof.shape
            else:
                if top_idx.size == 0:
                    xprof = x
                    rowsx,_ = xprof.shape
                    temp = np.zeros((rows,1))+np.nan
                    temp[:rowsx] = xprof
                    prof = np.hstack((prof,temp))
                else:
                    xprof = x[:top_idx]
                    prof = np.hstack((prof,xprof))
            n += 1
        except IndexError:
            pass
    meanx = np.expand_dims(np.nanmean(prof,axis=1),axis=1)
    y2 = y[:top_idx]
    ax.plot(meanx,y2,color=color,lw=3)
    ax.text(0.05,0.9,ax.get_gid().format(n),
                transform=ax.transAxes,
                fontsize=15)
     
    ax.set_xlim([-4,4])
    ax.set_ylim([0,2000])
    ax.axvline(0,color='k',linestyle=':',lw=3)
    
axes[0].set_ylabel('Altitude MSL [m]')
axes[6].set_xlabel(r'$N_{m}^{2} [x10^{-4} s^{-2}]$')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)

plt.subplots_adjust(hspace=0.07,wspace=0.05)

#plt.show()

fname='/home/raul/Desktop/fig_N2_7-storm.png'
plt.savefig(fname, dpi=150, format='png',papertype='letter',
            bbox_inches='tight')
