#----------------------------------------------------------------------
#   A library of postprocessing routines
#
#	Author	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-09-04
#----------------------------------------------------------------------

import numpy as np
import astropy.units as u
from . import dataset

def orbit_averaged_rate_of_inclination_change(p, t0=None, t1=None, simpson=False):
    """ Calculate the orbit average of the rate of change of inclination
    Assume p to be a cast.particle.Planet
    t0 and t1 are the boundaries of the integration interval
    in multiples of orbital periods """
    Torb = p["Torb"][0]
    if t0 is None:
        tmin = p['time'][0]
    else:
        tmin = t0*Torb
    if t1 is None:
        # Extract the last possible time such that tmax - tmin is a multiple of Torb
        b = p['time'][-1].to(Torb).value
        a = tmin.to(Torb).value
        tmax = (b-a) - (b-a)%1 + a
        tmax = tmax*Torb
    else:
        tmax = t1*Torb

    try:
        didt = p.data['didt']
    except KeyError:
        # Get forces
        Fx = p["a1"].data;
        Fy = p["a2"].data;
        Fz = p["a3"].data;
        # get orbital parameters
        a = p["a"].data
        r = a # for circular orbits
        e = p["e"].data
        i = p["i"].data
        # Calculate anomalies
        try:
            xi = p["xi"].data
        except KeyError:
            xi = p["Periastron"].data + p["TrueAnomaly"].data
        # sins and cosins
        cos_xi = np.cos(xi);
        sin_xi = np.sin(xi);
        cos_i = np.cos(i);
        sin_i = np.sin(i);
        # Calculate projections
        x = p['x1'].data
        y = p['x2'].data
        z = p['x3'].data
        vx = p['v1'].data
        vy = p['v2'].data
        vz = p['v3'].data

        eNx = y*vz - z*vy
        eNy = z*vx - x*vz
        eNz = x*vy - y*vx

        eNlen = np.sqrt(eNx**2 + eNy**2 + eNz**2)
        eNx /= eNlen
        eNy /= eNlen
        eNz /= eNlen

        FN = Fx*eNx + Fy*eNy + Fz*eNz
        
        H = eNlen  # specific angular momentum

        didt = r*FN*cos_xi/H*u.rad;
    
        didt = dataset.TimeSeries(name = "didt", time = p['time'], data = didt)
        p.data['didt'] = didt

    spl_time, spl_data = didt.between(tmin=tmin-Torb/20, tmax=tmax+Torb/20)

    X = spl_time.si.value
    Y = spl_data.si.value
    Xunit = spl_time.unit
    Yunit = spl_data.unit
    Norbits = (tmax - tmin).to('s')/Torb.to('s')

    if simpson:
        from scipy import integrate

        simps_time, simps_data = didt.between(tmin=tmin, tmax=tmax)
        rate = integrate.simps(simps_time.si.value, simps_data.si.value)/Norbits/Torb
        rate = (rate*spl_time.unit.si*spl_data.unit.si).decompose().to(u.rad/Torb)

    else:
        from scipy.interpolate import InterpolatedUnivariateSpline
        try:
            # Get the spline with the extended interval
            spl = InterpolatedUnivariateSpline(X, Y)
            # Calculate the integral with the initial integral
            rate = spl.integral(tmin.si.value, tmax.si.value)/Norbits/Torb
            rate = (rate*spl_time.unit.si*spl_data.unit.si).decompose().to(u.rad/Torb)
        except Exception:
            print('Got an exception for i = {}'.format(p['i'][0].to(u.degree)))
            rate = - 1e-10*u.rad/Torb

    return rate
