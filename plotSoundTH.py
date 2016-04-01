"""
    Plot NOAA balloon soundings as a
    time-height section.
    Files have extension tsv.

    Raul Valenzuela
    September, 2015
"""


import matplotlib.pyplot as plt
import numpy as np
import sounding as so

from scipy.ndimage.filters import gaussian_filter


# homedir = '/Users/raulv/Documents'
homedir = '/home/rvalenzuela'

usr_case = None


file_sound, usc = so.get_sounding_files(usr_case, homedir=homedir)

''' raw soundings vertically-interpolated '''
# # soundarray,_,_,_,_ = get_raw_array('thetaeq', file_sound)
out = so.get_raw_array('bvf_moist', file_sound)
soundarray, _, _, y, x, raw_dates = out
title = 'BVFm raw'
# make_imshow(soundarray,title,x,y,raw_dates)

''' time-interpolated '''
# soundarray2,_,_ = get_interp_array('u',files=file_sound)
# soundarray2,_,_ = get_interp_array('v',files=file_sound)
# soundarray2,_,_ = get_interp_array('DD',files=file_sound)
# soundarray2,_,_ = get_interp_array('thetaeq',files=file_sound)
out = so.get_interp_array('bvf_moist', files=file_sound)
soundarray2, hgt, timestamp, raw_dates = out

# make_imshow(soundarray2,'')

# make_contourf(soundarray2,'unfiltered array')

'''smooth field, sigma=2 seems good for BVF'''
sigma = 2
sound_filtered = gaussian_filter(soundarray2, sigma, mode='nearest')

x = timestamp
y = hgt
# array = sound_filtered
array = soundarray


if usc == str(3):
    ''' stable '''
    # st=np.datetime64('2001-01-23T16:00')
    # en=np.datetime64('2001-01-23T22:00')
    ''' unstable '''
    st = np.datetime64('2001-01-23T22:00')
    en = np.datetime64('2001-01-24T04:00')
elif usc == str(6):
    '''unstable '''
    st = np.datetime64('2001-02-11T02:20')
    en = np.datetime64('2001-02-11T06:00')
    ''' stable '''
    # st=np.datetime64('2001-02-11T06:00')
    # en=np.datetime64('2001-02-11T09:00')
elif usc == str(8):
    st = np.datetime64('2003-01-12T16:40')
    en = np.datetime64('2003-01-13T01:00')
elif usc == str(9):
    st = np.datetime64('2003-01-22T17:00')
    en = np.datetime64('2003-01-23T01:40')
elif usc == str(13):
    st = np.datetime64('2004-02-17T17:00')
    en = np.datetime64('2004-02-18T00:00')

'''**** CASE 14 NEEDS FURTHER QC ****'''

''' case 07 - P3 leg04 '''
st = np.datetime64('2001-02-17T17:20')
en = np.datetime64('2001-02-17T18:00')

title = 'BVFm with gaussian filter (sigma='+str(sigma)+')'
so.make_imshow(array, title, x, y, raw_dates)
# make_imshow(array,title,x,y,raw_dates,time=[st,en])
# make_imshow(array,title,x,y,None,vertical=[200,1200])
# make_imshow(array,title,x,y,None,vertical=[1200,4000])

# title = 'BVFm with gaussian filter (sigma='+str(sigma)+')'
# so.make_contourf(array, title, x, y, raw_dates)
# make_contourf(array,title,x,y,raw_dates,time=[st,en])
# so.make_contourf(array, title, x, y, raw_dates,
#                  vertical=[10, 1500], time=[st, en])
# make_contourf(array,title,x,y,raw_dates,
#     vertical=[200,3000],time=[st,en])
# make_contourf(array,title,x,y,raw_dates,
#     vertical=[1200,4000])

# array = so.make_statistical(sound_filtered, x, y,
#                             vertical=[10, 4000], time=[st, en])
# array = so.make_statistical(sound_filtered, x, y,
#                             vertical=[10, 1500], time=[st, en])

# array = make_statistical(sound_filtered, x, y,
#                          vertical=[200, 3000], time=[st, en])
# array = make_statistical(sound_filtered, x, y,
#                          vertical=[200, 1200])
# array = make_statistical(sound_filtered, x, y,
#                          vertical=[200, 1200], time=[st, en])
# array = make_statistical(sound_filtered, x, y,
#                          vertical=[1200, 4000])

# plt.show()
plt.show(block=False)
