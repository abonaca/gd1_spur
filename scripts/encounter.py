from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.constants import G
import gala.coordinates as gc

import scipy.optimize

import interact
import myutils

#import pickle

mw_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vsun = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vsun0 = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

gc_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 0.1*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vgc = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vgc0 = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

def morphology(seed=425, th=120):
    """Show results of an encounter in a log potential"""
    
    # impact parameters
    M = 1e8*u.Msun
    B = 19.85*u.kpc
    V = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.1*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 3000
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xr = 20*u.kpc + np.random.randn(Nstar)*0.02*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)

    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(12,8), sharex=True)
    
    c_init = mpl.cm.Blues_r(1)
    c_fin0 = mpl.cm.Blues_r(0.5)
    c_fin = mpl.cm.Blues_r(0.2)
    
    eta = coord.Angle(np.arctan2(np.sqrt(stream['x'][0].to(u.kpc).value**2 + stream['x'][1].to(u.kpc).value**2),xr.to(u.kpc).value)*u.rad)
    xi = np.arctan2(stream['x'][1].to(u.kpc).value, stream['x'][0].to(u.kpc).value)
    xi = coord.Angle((xi - np.median(xi))*u.rad)
    
    vlabel = ['x', 'y', 'z']
    
    for i in range(3):
        plt.sca(ax[i])
        im = plt.scatter(xi.deg, eta.deg, c=stream['v'][i].value, s=20)
        
        plt.xlim(-60, 50)
        plt.ylim(55, 35)
        plt.gca().set_aspect('equal')
        
        if i==2:
            plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('$\phi_2$ [deg]')
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.ylabel('$V_{{{}}}$ [km s$^{{-1}}$]'.format(vlabel[i]))
    
    plt.tight_layout()

def vel_difference(seed=425, th=120):
    """"""
    
    # impact parameters
    M = 1e8*u.Msun
    B = 19.85*u.kpc
    V = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.1*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 3000
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xr = 20*u.kpc + np.random.randn(Nstar)*0.02*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)

    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(12,9), sharex=True)
    
    c_init = mpl.cm.Blues_r(1)
    c_fin0 = mpl.cm.Blues_r(0.5)
    c_fin = mpl.cm.Blues_r(0.2)
    
    eta = coord.Angle(np.arctan2(np.sqrt(stream['x'][0].to(u.kpc).value**2 + stream['x'][1].to(u.kpc).value**2),xr.to(u.kpc).value)*u.rad)
    xphi_per = np.arctan2(stream['x'][1].to(u.kpc).value, stream['x'][0].to(u.kpc).value)
    xi = coord.Angle((xphi_per - np.median(xphi_per))*u.rad)
    
    vx_circ = -np.sin(xphi_per) * Vh
    vy_circ = np.cos(xphi_per) * Vh
    vz_circ = vx_circ * 0

    vlabel = ['x', 'y', 'z']
    vvalue = [vx_circ, vy_circ, vz_circ]
    
    plt.sca(ax[0])
    plt.plot(xi.deg, eta.deg, 'ko', mec='none')
    plt.xlim(-60, 50)
    plt.ylim(55, 35)
    plt.gca().set_aspect('equal')
    plt.ylabel('$\phi_2$ [deg]')
    
    for i in range(3):
        plt.sca(ax[i+1])
        #im = plt.scatter(xi.deg, stream['v'][i].value, c=eta.deg, s=20)
        plt.plot(xi.deg, stream['v'][i] - vvalue[i], 'ko', mec='none')
        
        plt.xlim(-60, 50)
        #plt.ylim(55, 35)
        #plt.gca().set_aspect('equal')
        
        if i==2:
            plt.xlabel('$\phi_1$ [deg]')
        #plt.ylabel('$\phi_2$ [deg]')
        
        #divider = make_axes_locatable(plt.gca())
        #cax = divider.append_axes("right", size="3%", pad=0.1)
        #plt.colorbar(im, cax=cax)
        plt.ylabel('$\Delta$ $V_{{{}}}$ [km s$^{{-1}}$]'.format(vlabel[i]))
    
    plt.tight_layout()


def sky(seed=425, th=150):
    """Project results of an encounter in a log potential on the sky"""
    
    # impact parameters
    M = 1e8*u.Msun
    B = 19.85*u.kpc
    V = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.1*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 3000
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 0*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xr = 20*u.kpc + np.random.randn(Nstar)*0.02*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    
    # generate stream model
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    # sky coordinates
    xgal = coord.Galactocentric(stream['x'], **observer)
    xeq = xgal.transform_to(coord.ICRS)
    veq_ = gc.vgal_to_hel(xeq, stream['v'], **vobs)
    veq = [None] * 3
    veq[0] = veq_[0].to(u.mas/u.yr)
    veq[1] = veq_[1].to(u.mas/u.yr)
    veq[2] = veq_[2].to(u.km/u.s)
    
    # rotate to native coordinate system
    R = find_greatcircle(xeq.ra.deg[::10], xeq.dec.deg[::10])
    xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
    xi = coord.Angle(xi*u.deg)
    
    # unperturbed stream
    par_perturb = np.array([0*M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream0 = {}
    stream0['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream0['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    # sky coordinates
    xgal0 = coord.Galactocentric(stream0['x'], **observer)
    xeq0 = xgal0.transform_to(coord.ICRS)
    veq0_ = gc.vgal_to_hel(xeq0, stream0['v'], **vobs)
    veq0 = [None] * 3
    veq0[0] = veq0_[0].to(u.mas/u.yr)
    veq0[1] = veq0_[1].to(u.mas/u.yr)
    veq0[2] = veq0_[2].to(u.km/u.s)
    
    # rotate to native coordinate system
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    ylims = [[-0.5, 0.5], [-0.5, 0.5], [-25,25]]
    color = '0.35'
    ms = 8
    
    # plotting
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(12,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms)
    
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    xeqs = [xeq.ra, xeq.dec, xeq.distance.to(u.kpc)]
    for i in range(3):
        plt.sca(ax[i+1])
        
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms)
        
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/spur_morphology_sky.png')



def find_greatcircle(ra_deg, dec_deg):
    """Save rotation matrix for a stream model"""
    
    #stream = stream_model(name, pparams0=pparams, dt=dt)
    
    ## find the pole
    #ra = np.radians(stream.obs[0])
    #dec = np.radians(stream.obs[1])
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    
    rx = np.cos(ra) * np.cos(dec)
    ry = np.sin(ra) * np.cos(dec)
    rz = np.sin(dec)
    r = np.column_stack((rx, ry, rz))
    #r = sph2cart(ra, dec)

    # fit the plane
    x0 = np.array([0, 1, 0])
    lsq = scipy.optimize.minimize(wfit_plane, x0, args=(r,))
    x0 = lsq.x/np.linalg.norm(lsq.x)
    ra0 = np.arctan2(x0[1], x0[0])
    dec0 = np.arcsin(x0[2])
    
    ra0 += np.pi
    dec0 = np.pi/2 - dec0

    # euler rotations
    R0 = myutils.rotmatrix(np.degrees(-ra0), 2)
    R1 = myutils.rotmatrix(np.degrees(dec0), 1)
    R2 = myutils.rotmatrix(0, 2)
    R = np.dot(R2, np.matmul(R1, R0))
    
    xi, eta = myutils.rotate_angles(ra_deg, dec_deg, R)
    
    # put xi = 50 at the beginning of the stream
    xi[xi>180] -= 360
    xi += 360
    xi0 = np.min(xi) - 50
    R2 = myutils.rotmatrix(-xi0, 2)
    R = np.dot(R2, np.matmul(R1, R0))
    xi, eta = myutils.rotate_angles(ra_deg, dec_deg, R)
    
    return R

def wfit_plane(x, r, p=None):
    """Fit a plane to a set of 3d points"""
    
    Np = np.shape(r)[0]
    if np.any(p)==None:
        p = np.ones(Np)
    
    Q = np.zeros((3,3))
    
    for i in range(Np):
        Q += p[i]**2 * np.outer(r[i], r[i])
    
    x = x/np.linalg.norm(x)
    lsq = np.inner(x, np.inner(Q, x))
    
    return lsq
