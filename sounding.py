import numpy as np
import os
from datetime import datetime, timedelta
# import sys

# import Thermodyn as thermo
import Meteoframes as mf
import pandas as pd

import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.mlab as mlab

from scipy.interpolate import interp1d
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from itertools import product


def make_statistical(sound_filtered, X, Y, **kwargs):

    vertical = False
    time = False
    if kwargs:
        for key, value in kwargs.iteritems():
            if key == 'vertical':
                vertical = True
                bot = np.where(Y == value[0])[0]
                top = np.where(Y == value[1])[0]
            if key == 'time':
                time = True
                st = np.where(X == value[0])[0]
                en = np.where(X == value[1])[0]

    ''' statistical analysis of bvf '''
    if vertical and time:
        bvf_sector = sound_filtered[bot:top, st:en]*1e4
    elif vertical:
        bvf_sector = sound_filtered[bot:top, :]*1e4
    elif time:
        bvf_sector = sound_filtered[:, st:en]*1e4
    else:
        bvf_sector = sound_filtered[:, :]*1e4

    ''' reshape to one column '''
    bvf_sector_rsh = np.reshape(bvf_sector, bvf_sector.size, 1)

    vmin = -2.
    vmax = 2.
    vdel = 0.5
    n, bins = np.histogram(
        bvf_sector_rsh, bins=np.arange(vmin, vmax+vdel, vdel), normed=True)
    n = np.append(n, 0.)
    prob = n*vdel
    print np.sum(prob)
    fig, ax = plt.subplots()
    ax.bar(bins, prob, width=0.5)
    mu = np.mean(bvf_sector_rsh)
    sigma = np.std(bvf_sector_rsh)
    res = vdel*0.1
    x = np.arange(-5., 5., res)
    y = mlab.normpdf(x, mu, sigma)
    print np.sum(y)*res
    l = ax.plot(x, y, 'r--', linewidth=3)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([0., 1.2])
    strfloat = '{:2.1f}'
    strint = '{:d}'
    ax.text(-2, 1, r'$\mu='+strfloat.format(mu) +
            ',\ \sigma='+'{:2.1f}'.format(sigma)+'$', size=18)
    plt.xlabel(r'$BVFm [\times10^{-4} s^{-2}]$')
    plt.ylabel('Normal PDF')
    bl = Y[bot][0]
    tl = Y[top][0]
    if bl == 10:
        bl = 0
    plt.title(r'$\mathrm{Histogram\ of\ BVFm}\ Layer: ' +
              strint.format(bl)+'-'+strint.format(tl)+'mAGL$')
    plt.draw()

    return bvf_sector_rsh


def make_contourf(array, title, X, Y, raw_dates, **kwargs):

    bvf_clevels = np.arange(-4e-4, 10e-4, 2e-4)

    vertical = False
    time = False
    if kwargs:
        for key, value in kwargs.iteritems():
            if key == 'vertical':
                vertical = True
                bot = value[0]
                top = value[1]
            if key == 'time':
                time = True
                st = value[0]
                en = value[1]

    ''' repeat last column so contourf can fill until X.size '''
    lastc = array[:, -1]
    lastc = np.reshape(lastc, (lastc.size, 1))
    array = np.hstack((array, lastc))

    ''' filtered contourf plot '''
    rows, cols = array.shape
    xx, yy = np.meshgrid(range(cols), range(rows))
    fig, ax = plt.subplots()
    cf = ax.contourf(xx, yy, array, levels=bvf_clevels)

    t = [pd.to_datetime(x - np.timedelta64(7, 'h')) for x in X]
    M = t[0].minute
    if M == 0:
        xticks = range(0, X.size, 3)
    elif M == 20:
        xticks = range(2, X.size, 3)
    elif M == 40:
        xticks = range(1, X.size, 3)

    # xticks=range(0,X.size,3)
    ax.set_xticks(xticks)
    date_fmt = '%d\n%H'
    t = [pd.to_datetime(x - np.timedelta64(7, 'h')) for x in X]
    xticklabs = [t[i].strftime(date_fmt) for i in xticks]
    ax.set_xticklabels(xticklabs)

    yticks = range(0, Y.size, 20)
    ax.set_yticks(yticks)
    yticklabs = [str(Y[j]-10) for j in yticks]
    ax.set_yticklabels(yticklabs)

    if time:
        st = np.where(X == st)[0]
        en = np.where(X == en)[0]
        ax.set_xlim([st, en+0.25])
    else:
        ax.set_xlim([0, X.size-0.5])

    if vertical:
        bot = np.where(Y == bot)[0]
        top = np.where(Y == top)[0]
        ax.set_ylim([bot, top])

    if raw_dates:
        xidx = [np.where(X == r)[0][0] for r in raw_dates]
        yidx = np.repeat([20], len(xidx))
        ax.plot(xidx, yidx, 'o', color='black', markersize=8)

    ax.invert_xaxis()
    plt.colorbar(cf)
    plt.subplots_adjust(left=0.125, right=0.98, top=0.95, bottom=0.12)
    plt.suptitle(title)
    plt.xlabel(r'$\Leftarrow$'+' Time [UTC]')
    plt.ylabel('Altitude AGL [m]')
    plt.draw()


def make_imshow(array, title, X, Y, raw_dates, **kwargs):

    vertical = False
    time = False
    if kwargs:
        for key, value in kwargs.iteritems():
            if key == 'vertical':
                vertical = True
                bot = value[0]
                top = value[1]
            if key == 'time':
                time = True
                st = value[0]
                en = value[1]

    fig, ax = plt.subplots()
    ''' extent values help to put ticks in the middle of pixel '''
    im = ax.imshow(array, interpolation='none', origin='top',
                   aspect='auto', vmin=-8e-4, vmax=8e-4,
                   extent=[-0.5, X.size-0.5, 1, 399],cmap='PuOr')

    t = [pd.to_datetime(x - np.timedelta64(7, 'h')) for x in X]
    M = t[0].minute
    if M == 0:
        xticks = range(0, X.size, 3)
    elif M == 20:
        xticks = range(2, X.size, 3)
    elif M == 40:
        xticks = range(1, X.size, 3)

    date_fmt = '%d\n%H'
    xticklabs = [t[i].strftime(date_fmt) for i in xticks]
    ax.set_xticklabels(xticklabs)
    ax.set_xticks(np.asarray(xticks))

    yticks = range(0, Y.size, 20)
    ax.set_yticks(yticks)
    yticklabs = [str(Y[j]-10) for j in yticks]
    ax.set_yticklabels(yticklabs)

    if time:
        st = np.where(X == st)[0]
        en = np.where(X == en)[0]
        ax.set_xlim([st, en])

    if vertical:
        bot = np.where(Y == bot)[0]
        top = np.where(Y == top)[0]
        ax.set_ylim([bot, top])

    if raw_dates:
        xidx = [np.where(X == r)[0][0] for r in raw_dates]
        yidx = np.repeat([20], len(xidx))
        ax.plot(xidx, yidx, 'o')

    ax.invert_xaxis()
    plt.subplots_adjust(left=0.125, right=0.98, top=0.95, bottom=0.12)
    plt.xlabel(r'$\Leftarrow$'+' Time [UTC]')
    plt.ylabel('Altitude AGL [m]')
    plt.suptitle(title)
    plt.colorbar(im)
    plt.draw()


def get_sounding_files(usr_case, homedir=None):
    ''' set directory and input files '''
    base_directory = homedir + '/BALLOON'
    if not usr_case:
        print base_directory
        usr_case = raw_input('\nIndicate case number (i.e. 1): ')
    case = 'case'+usr_case.zfill(2)
    casedir = base_directory+'/'+case
    out = os.listdir(casedir)
    file_sound = []
    for f in out:
        if f[-3:] == 'tsv':
            file_sound.append(casedir+'/'+f)

    return file_sound, usr_case


def get_raw_array(soundvar, file_sound):

    file_sound.sort()

    ''' height grid with 10 m resolution'''

    var = []  # 2D list
    timestamps = []
    top_limit = 10000  # [m]
    hgtgrid = np.asarray(range(10, 4010, 10))
    for f in file_sound:

        ''' sounding timestamps approached to
        the nearest 20 minute grid point'''
        fname = os.path.basename(f)
        raw_date = fname[:-4].split('_')[-2:]
        y = raw_date[0][:4]
        m = raw_date[0][4:6]
        d = raw_date[0][6:8]
        hh = int(raw_date[1][0:2])
        mm = int(raw_date[1][2:4])
        nearest = round_to_nearest(mm, 20)
        if nearest == 60:
            hh += 1
            mm = 0
        else:
            mm = nearest
        if hh == 24:
            d = str(int(d)+1)
            hh = 0
        raw_date = y+'-'+m+'-'+d+'T' + \
            str(hh).zfill(2)+':'+str(mm).zfill(2)+':'+str(0).zfill(2)
        timestamps.append(np.datetime64(raw_date))

        ''' variable '''
        df = mf.parse_sounding2(f)
        # print df
        sv = df[soundvar][df.index < top_limit].values

        Z = df.index[df.index < top_limit].values
        ''' interpolate to a common vertical grid '''
        f = interp1d(Z, sv)
        svinterp = np.asarray([])
        for h in hgtgrid:
            try:
                svinterp = np.append(svinterp, f(h))
            except ValueError:
                svinterp = np.append(svinterp, np.nan)

        ''' take care of vertical nan values in raw soundings;
        more gaps can show up if the top of the vertical grid
        is increased (i.e. higher balloon altitudes show more
        missing data)
        '''
        if np.any(np.isnan(svinterp)):
            nanidx = np.where(np.isnan(svinterp))[0]
            diff = np.diff(nanidx)
            idxjump = np.where(diff > 1)[0]
            ngaps = len(idxjump) + 1
            if ngaps > 1 and nanidx[-1] != svinterp.size-1:
                gapidx = np.split(nanidx, idxjump+1)
                for g in gapidx:
                    first = g[0]
                    last = g[-1]
                    ''' pick good values between nans and
                    make a linear interpolation (hopefully there
                    few nans) '''
                    if last+1 == svinterp.size:
                        x = [hgtgrid[first-1], hgtgrid[last]]
                        y = [svinterp[first-1], svinterp[last]]
                    elif first == 0:
                        x = [hgtgrid[first], hgtgrid[last+1]]
                        y = [svinterp[first], svinterp[last+1]]
                    else:
                        x = [hgtgrid[first-1], hgtgrid[last+1]]
                        y = [svinterp[first-1], svinterp[last+1]]
                    f = interp1d(x, y)
                    svinterp[g] = [f(h) for h in hgtgrid[g]]

            elif ngaps == 1 and nanidx[-1] != svinterp.size-1:
                first = nanidx[0]
                last = nanidx[-1]
                if last+1 == svinterp.size:
                    x = [hgtgrid[first-1], hgtgrid[last]]
                    y = [svinterp[first-1], svinterp[last]]
                elif first == 0:
                    x = [hgtgrid[first], hgtgrid[last+1]]
                    y = [svinterp[first], svinterp[last+1]]
                else:
                    x = [hgtgrid[first-1], hgtgrid[last+1]]
                    y = [svinterp[first-1], svinterp[last+1]]
                f = interp1d(x, y)
                svinterp[nanidx] = [f(h) for h in hgtgrid[nanidx]]
            var.append(svinterp)
        else:
            ''' add column to list '''
            var.append(svinterp)

    ''' time grid with 20 minute spacing'''
    t1 = timestamps[0]-np.timedelta64(20, 'm')
    t2 = timestamps[-1]+np.timedelta64(20, 'm')
    dates = np.arange(t1, t2, dtype='datetime64[20m]')

    ''' get index of each sounding date '''
    tidx = []
    for t in timestamps:
        tidx.append(np.where(dates == t)[0][0])

    ''' assign soundings to 2D array '''
    soundarray = np.full((hgtgrid.size, dates.size), np.nan)
    for n, i in enumerate(tidx):
        soundarray[:, i] = var[n]

    return soundarray, var, tidx, hgtgrid, dates, timestamps


def get_interp_array(soundvar, homedir=None, **kwargs):
    ''' interpolate gaps in time
    '''

    for key, value in kwargs.iteritems():
        if key == 'case':
            file_sound = get_sounding_files(value,
                                            homedir=homedir)[0]
        elif key == 'files':
            file_sound = value

    out = get_raw_array(soundvar, file_sound)
    soundarray, var, tidx, hgtgrid, dates, raw_dates = out

    ''' create column variables x,y,z with all available data to feed Rbf '''
    x = np.repeat(tidx, len(hgtgrid), axis=0)
    x = x[:, np.newaxis]

    y = hgtgrid[:, np.newaxis].T
    y = np.tile(y, len(tidx))
    y = np.reshape(y, (y.size, 1))

    z = np.asarray(var)
    z = np.reshape(z, (z.size, 1))

    ''' when soundings have nans '''
    nans = np.isnan(z)
    nonnan = np.where(~nans)[0]

    ''' 2D interpolation '''
    rbf = Rbf(x[nonnan], y[nonnan], z[nonnan])
    soundarray2 = np.copy(soundarray)
    for d in range(len(dates)):
        if d not in tidx:
            xi = np.repeat(d, len(hgtgrid))
            zi = rbf(xi, hgtgrid)
            soundarray2[:, d] = zi

    ''' fill remaining gaps '''
    if np.any(nans):
        soundarray2 = fill_nan_gaps(soundarray2)

    return soundarray2, hgtgrid, dates, raw_dates


def fill_nan_gaps(array):
    """ search nearest neighbors """
    a, b = array.shape
    coords = list(product(range(a), range(b)))
    tree = cKDTree(coords)
    neigh = 9

    ''' fills nans by averaging closest neighbors'''
    gx, gy = np.where(np.isnan(array))
    array_col = np.reshape(array, (a*b, 1))
    for i, j in zip(gx, gy):
        dist, idx = tree.query([i, j], k=neigh)
        new_value = np.nanmean(array_col[idx])
        array[i, j] = new_value
    return array


def round_to_nearest(num, base):
    ''' see tzaman answer in stackoverflow '''
    n = num + (base//2)
    return n - (n % base)


def get_dates(start, end, delta):

    dates = []
    if delta == timedelta(hours=0):
        dates.append(start)
        return dates
    else:
        foo = start
        dates.append(start)
        while foo <= end:
            foo += delta
            dates.append(foo)
        return dates


def parse_dates(file_sound):

    raw_dates = []
    for f in file_sound:
        fs = os.path.basename(f)[:-4]
        raw_dates.append(datetime.strptime(fs, '%Y%m%d_%H%M%S'))

    return raw_dates
