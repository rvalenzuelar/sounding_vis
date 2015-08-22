
file_sounds


def parse_dataframe(file_met):

	dates_col=[0,1,2]
	dates_fmt='%Y %j %H%M'

	''' read the csv file '''
	dframe = pd.read_csv(file_met,header=None)

	''' parse date columns into a single date col '''
	raw_dates=dframe.ix[:,dates_col]
	raw_dates.columns=['Y','j','HHMM']
	raw_dates['HHMM'] = raw_dates['HHMM'].apply(lambda x:'{0:0>4}'.format(x))
	raw_dates=raw_dates.apply(lambda x: '%s %s %s' % (x['Y'],x['j'],x['HHMM']), axis=1)
	dates=raw_dates.apply(lambda x: datetime.strptime(x, dates_fmt))

	''' make meteo df, assign datetime index, and name columns '''
	meteo=dframe.ix[:,index_field]
	meteo.index=dates
	meteo.columns=name_field

	''' make field with hourly acum precip '''
	hour=pd.TimeGrouper('H')
	preciph = meteo.precip.groupby(hour).sum()
	meteo = meteo.join(preciph, how='outer', rsuffix='h')

	''' add thermodynamics '''
	theta = thermo.theta(C=meteo.temp,hPa=meteo.press)
	thetaeq = thermo.theta_equiv(C=meteo.temp,hPa=meteo.press)
	meteo.loc[:,'theta'] = pd.Series(theta,index=meteo.index)	
	meteo.loc[:,'thetaeq'] = pd.Series(thetaeq,index=meteo.index)	

	''' assign metadata (prototype, not really used) '''
	units = {'press':'mb', 'temp':'C', 'rh':'%', 'wspd':'m s-1', 'wdir':'deg', 'precip':'mm', 'mixr': 'g kg-1'}
	agl = {'press':'NaN', 'temp':'10 m', 'rh':'10 m', 'wspd':'NaN', 'wdir':'NaN', 'precip':'NaN', 'mixr': 'NaN'}
	for n in name_field:
		meteo[n].units=units[n]
		meteo[n].agl=agl[n]
		meteo[n].nan=-9999.999
		meteo[n].sampling_freq='1 minute'	

	return meteo