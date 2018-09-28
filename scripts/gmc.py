from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gd1 import *

def disk_crossing_times(pot='mw', T=1*u.Gyr, dt=0.01*u.Myr, verbose=True, graph=True):
    """Return times when the gap crosses the disk"""
    
    # load gap at the present
    pkl = pickle.load(open('../data/gap_present_{}.pkl'.format(pot), 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    n_steps = np.int64(T/dt)
    t = np.arange(0, T.to(u.Myr).value + dt.to(u.Myr).value, dt.to(u.Myr).value)

    ham = ham_mw
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    
    # sign change == disk crossing
    asign = np.sign(fit_orbit.z)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(bool)
    times = t[signchange]
    
    if verbose:
        print(times)
        print(fit_orbit.cylindrical.rho[signchange])
        print(fit_orbit.z[signchange])
    
    if graph:
        plt.close()
        plt.plot(t, fit_orbit.z, 'k-')
        
        for t_ in times:
            plt.axvline(t_, color='salmon')
        
        plt.xlabel('Time [Myr]')
        plt.ylabel('z [kpc]')
        plt.tight_layout()
    
    return times*u.Myr
