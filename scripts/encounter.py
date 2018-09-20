from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.constants import G
from astropy.io import fits

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp

import scipy.optimize

import interact
import myutils

import pickle

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


def sky(seed=425, th=150, old=False):
    """Project results of an encounter in a log potential on the sky"""
    
    # impact parameters
    M = 3e7*u.Msun
    B = 19.95*u.kpc
    #B = 20.08*u.kpc
    V = 190*u.km/u.s
    phi = coord.Angle(0*u.deg)
    th = 150
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    old_label = ''
    np.random.seed(seed)
    observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    if old:
        old_label = '_old_up'
        observer = {'z_sun': -2000.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 50*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
        vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0,0,0]*u.km/u.s}
        
        # impact parameters
        M = 3e7*u.Msun
        B = 20.06*u.kpc
        V = 190*u.km/u.s
        phi = coord.Angle(0*u.deg)
        th = 155
        theta = coord.Angle(th*u.deg)
        Tenc = 0.01*u.Gyr
        T = 0.55*u.Gyr
        dt = 0.05*u.Myr
        #dt = 1*u.Myr
        rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 1400
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xphi0 = np.linspace(-0.1*np.pi, 0.1*np.pi, 1000)
    xphi1 = np.linspace(-0.28*np.pi, -0.1*np.pi, 200)
    xphi2 = np.linspace(0.1*np.pi, 0.32*np.pi, 200)
    xphi = np.concatenate([xphi1, xphi0, xphi2])
    
    xr = 20*u.kpc + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh# * 0.94
    vy = np.sin(xphi) * Vh #* 0.97
    vz = vx * 0
    # closest to impact
    ienc = np.argmin(np.abs(x))
    
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
    R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    # place gap at xi~0
    xioff = xi0[ienc]
    xi0 -= xioff
    
    xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
    xi = coord.Angle(xi*u.deg)
    xi -= xioff
    
    vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    ylims = [[-0.5, 0.5], [-0.5, 0.5], [-25,25]]
    color = '0.35'
    ms = 4
    
    # plotting
    plt.close()
    fig, ax = plt.subplots(5,1,figsize=(12,12), sharex=True)
    
    plt.sca(ax[0])
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    plt.scatter(g['phi1']+40, g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1)
    
    plt.xlim(-45,45)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    plt.ylabel('$\phi_1$ [deg]')
    
    plt.sca(ax[1])
    plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms)
    
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    xeqs = [xeq.ra, xeq.dec, xeq.distance.to(u.kpc)]
    for i in range(3):
        plt.sca(ax[i+2])
        
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms)
        
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/spur_morphology_sky{}.png'.format(old_label))
    
def sky_observed(seed=425, th=150, old=False):
    """Project results of an encounter in a log potential on the sky"""
    
    # impact parameters
    M = 3e7*u.Msun
    #M = 6e7*u.Msun
    B = 19.95*u.kpc
    V = 190*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    old_label = ''
    
    if old:
        old_label = '_old'
        
        # impact parameters
        M = 5e7*u.Msun
        B = 19.8*u.kpc
        V = 210*u.km/u.s
        phi = coord.Angle(0*u.deg)
        th = 150
        theta = coord.Angle(th*u.deg)
        Tenc = 0.05*u.Gyr
        T = 2*u.Gyr
        dt = 0.1*u.Myr
        #dt = 1*u.Myr
        rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 1400
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    alt_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': -45*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xphi0 = np.linspace(-0.1*np.pi, 0.1*np.pi, 1000)
    xphi1 = np.linspace(-0.28*np.pi, -0.1*np.pi, 200)
    xphi2 = np.linspace(0.1*np.pi, 0.32*np.pi, 200)
    xphi = np.concatenate([xphi1, xphi0, xphi2])
    
    xr = 20*u.kpc + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    # closest to impact
    ienc = np.argmin(np.abs(x))
    
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
    
    # alternative sky coordinates
    xgal_alt = coord.Galactocentric(stream['x'], **alt_observer)
    xeq_alt = xgal_alt.transform_to(coord.ICRS)
    veq_alt_ = gc.vgal_to_hel(xeq_alt, stream['v'], **vobs)
    veq_alt = [None] * 3
    veq_alt[0] = veq_alt_[0].to(u.mas/u.yr)
    veq_alt[1] = veq_alt_[1].to(u.mas/u.yr)
    veq_alt[2] = veq_alt_[2].to(u.km/u.s)
    
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
    
    # alternative sky coordinates
    xgal0_alt = coord.Galactocentric(stream0['x'], **alt_observer)
    xeq0_alt = xgal0_alt.transform_to(coord.ICRS)
    veq0_alt_ = gc.vgal_to_hel(xeq0_alt, stream0['v'], **vobs)
    veq0_alt = [None] * 3
    veq0_alt[0] = veq0_alt_[0].to(u.mas/u.yr)
    veq0_alt[1] = veq0_alt_[1].to(u.mas/u.yr)
    veq0_alt[2] = veq0_alt_[2].to(u.km/u.s)
    
    # rotate to native coordinate system
    R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    # place gap at xi~0
    xioff = xi0[ienc]
    xi0 -= xioff
    
    xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
    xi = coord.Angle(xi*u.deg)
    xi -= xioff
    
    # alternative observer
    R_alt = find_greatcircle(xeq0_alt.ra.deg[::10], xeq0_alt.dec.deg[::10])
    xi0_alt, eta0_alt = myutils.rotate_angles(xeq0_alt.ra, xeq0_alt.dec, R_alt)
    xi0_alt = coord.Angle(xi0_alt*u.deg)
    
    # place gap at xi~0
    xioff_alt = xi0_alt[ienc]
    xi0_alt -= xioff_alt
    
    xi_alt, eta_alt = myutils.rotate_angles(xeq_alt.ra, xeq_alt.dec, R_alt)
    xi_alt = coord.Angle(xi_alt*u.deg)
    xi_alt -= xioff_alt
    

    # observed gd1
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    
    vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    ylims = [[-1.5, 1.5], [-1.5, 1.5], [-30,30]]
    color = '0.35'
    ms = 4
    alpha = 0.7
    
    # plotting
    plt.close()
    fig, ax = plt.subplots(2,4,figsize=(17,8), sharex=True, sharey='col')
    
    plt.sca(ax[0][0])
    plt.plot(xi.wrap_at(wangle), eta, '.', mec='none', color=color, ms=ms, label='Simulated GD-1')
    
    #plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    plt.xlim(-20,20)
    plt.ylim(-5,5)
    
    plt.sca(ax[1][0])
    plt.plot(xi_alt.wrap_at(wangle), eta_alt, '.', mec='none', color=color, ms=ms, label='Simulated GD-1')
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    plt.xlim(-20,20)
    plt.ylim(-5,5)
    
    xeqs = [xeq.ra, xeq.dec, xeq.distance.to(u.kpc)]
    dv = []
    dv_alt = []
    for i in range(3):
        plt.sca(ax[0][i+1])
        
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        dv += [veq[i]-vexp]
        plt.plot(xi.wrap_at(wangle), dv[i], '.', mec='none', color=color, ms=ms)
        
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])
        
        plt.sca(ax[1][i+1])
        # interpolate expected kinematics from an unperturbed stream
        vexp_alt = np.interp(xi_alt.wrap_at(wangle), xi0_alt.wrap_at(wangle), veq0_alt[i].value) * veq0_alt[i].unit
        dv_alt += [veq_alt[i]-vexp_alt]
        plt.plot(xi_alt.wrap_at(wangle), dv_alt[i], '.', mec='none', color=color, ms=ms)
        
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])
        plt.xlabel('$\phi_1$ [deg]')
    
    # find closest model star to the gd-1 stars
    Ngd1 = len(g)
    p = np.array([g['phi1']+40, g['phi2']])
    q = np.array([xi.wrap_at(wangle).to(u.deg).value, eta])
    idmin = np.empty(Ngd1, dtype='int')
    
    for i in range(Ngd1):
        dist = np.sqrt((p[0,i]-q[0])**2 + (p[1,i]-q[1])**2)
        idmin[i] = np.argmin(dist)

    # mask stream, mask spur
    onstream_mask = ((g['phi1']<-30.5) & (g['phi1']>-35.5) & (g['phi2']>-0.2) & (g['phi2']<0.2))
    spur_mask = ((g['phi1']<-30.5) & (g['phi1']>-35.5) & (g['phi2']>1) & (g['phi2']<1.4))
    all_mask = np.ones(Ngd1, dtype='bool')
    
    # plot scaled data uncertainties on model pm drawn from a corresponding obs uncertainty
    np.random.seed(seed+1)
    fgaia = np.sqrt(2/5)
    print(2/5, fgaia)
    phi1 = xi[idmin].wrap_at(wangle).to(u.deg).value
    phi2 = eta[idmin]
    pmra = dv[0][idmin] + g['pmra_error']*u.mas/u.yr*np.random.randn(Ngd1) * fgaia
    pmdec = dv[1][idmin] + g['pmdec_error']*u.mas/u.yr*np.random.randn(Ngd1) * fgaia
    
    colors = ['tab:red', 'tab:blue', '0.4']
    labels = ['Stream', 'Spur']
    labels = ['Gaia DR4', '']
    
    for e, mask in enumerate([onstream_mask, spur_mask]):
        plt.sca(ax[0][0])
        plt.plot(phi1[mask], phi2[mask], 'o', color=colors[e], mec='none', alpha=alpha, label=labels[e])
        
        plt.sca(ax[0][1])
        plt.errorbar(phi1[mask], pmra[mask].value, yerr=g['pmra_error'][mask]*fgaia, fmt='o', color=colors[e], mec='none', alpha=alpha)
        
        plt.sca(ax[0][2])
        plt.errorbar(phi1[mask], pmdec[mask].value, yerr=g['pmdec_error'][mask]*fgaia, fmt='o', color=colors[e], mec='none', alpha=alpha)
        
        print(np.sqrt(np.sum(g['pmra_error'][mask]**2))/np.sum(mask))
        print(np.sqrt(np.sum(g['pmdec_error'][mask]**2))/np.sum(mask))

    Nfield = 2
    p2 = np.array([np.array([-32.77,-32.77])+40, [1.167,0]])
    q = np.array([xi.wrap_at(wangle).to(u.deg).value, eta])
    idmin2 = np.empty(Nfield, dtype='int')
    
    for i in range(Nfield):
        dist = np.sqrt((p2[0,i]-q[0])**2 + (p2[1,i]-q[1])**2)
        idmin2[i] = np.argmin(dist)
    
    pmerr = np.array([0.0848, 0.0685])
    
    np.random.seed(seed+2)
    phi1 = xi[idmin2].wrap_at(wangle).to(u.deg).value
    phi2 = eta[idmin2]
    pmra = dv[0][idmin2].value + pmerr*np.random.randn(Nfield)
    pmdec = dv[1][idmin2].value + pmerr*np.random.randn(Nfield)
    
    plt.sca(ax[0][0])
    plt.errorbar(phi1, phi2, color='k', fmt='o', label='HST')
    
    plt.sca(ax[0][1])
    plt.errorbar(phi1, pmra, yerr=pmerr, color='k', fmt='o')
    
    plt.sca(ax[0][2])
    plt.errorbar(phi1, pmdec, yerr=pmerr, color='k', fmt='o')
    
    
    ##############
    # alt observer
    
    # find closest model star to the gd-1 stars
    Ngd1 = len(g)
    p = np.array([g['phi1']+40, g['phi2']])
    q = np.array([xi_alt.wrap_at(wangle).to(u.deg).value, eta_alt])
    idmin = np.empty(Ngd1, dtype='int')
    
    for i in range(Ngd1):
        dist = np.sqrt((p[0,i]-q[0])**2 + (p[1,i]-q[1])**2)
        idmin[i] = np.argmin(dist)

    # mask stream, mask spur
    onstream_mask = ((g['phi1']<-30.5) & (g['phi1']>-35.5) & (g['phi2']>-0.2) & (g['phi2']<0.2))
    spur_mask = ((g['phi1']<-30.5) & (g['phi1']>-35.5) & (g['phi2']>1) & (g['phi2']<1.4))
    all_mask = np.ones(Ngd1, dtype='bool')
    
    # plot scaled data uncertainties on model pm drawn from a corresponding obs uncertainty
    #np.random.seed(seed+3)
    phi1 = xi_alt[idmin].wrap_at(wangle).to(u.deg).value
    phi2 = eta_alt[idmin]
    pmra = dv_alt[0][idmin] + g['pmra_error']*u.mas/u.yr*np.random.randn(Ngd1) * fgaia
    pmdec = dv_alt[1][idmin] + g['pmdec_error']*u.mas/u.yr*np.random.randn(Ngd1) * fgaia
    
    colors = ['tab:red', 'tab:blue', '0.4']
    labels = ['Gaia DR4', '']
    
    for e, mask in enumerate([onstream_mask, spur_mask]):
        plt.sca(ax[1][0])
        plt.plot(phi1[mask], phi2[mask], 'o', color=colors[e], mec='none', alpha=alpha, label=labels[e])
        
        plt.sca(ax[1][1])
        plt.errorbar(phi1[mask], pmra[mask].value, yerr=g['pmra_error'][mask]*fgaia, fmt='o', color=colors[e], mec='none', alpha=alpha)
        
        plt.sca(ax[1][2])
        plt.errorbar(phi1[mask], pmdec[mask].value, yerr=g['pmdec_error'][mask]*fgaia, fmt='o', color=colors[e], mec='none', alpha=alpha)
        
    Nfield = 2
    p2 = np.array([np.array([-32.77,-32.77])+40, [1.167,0]])
    q = np.array([xi_alt.wrap_at(wangle).to(u.deg).value, eta_alt])
    idmin2 = np.empty(Nfield, dtype='int')
    
    for i in range(Nfield):
        dist = np.sqrt((p2[0,i]-q[0])**2 + (p2[1,i]-q[1])**2)
        idmin2[i] = np.argmin(dist)
    
    pmerr = np.array([0.11, 0.08])
    
    np.random.seed(seed+6)
    phi1 = xi_alt[idmin2].wrap_at(wangle).to(u.deg).value
    phi2 = eta_alt[idmin2]
    pmra = dv_alt[0][idmin2].value + pmerr*np.random.randn(Nfield)
    pmdec = dv_alt[1][idmin2].value + pmerr*np.random.randn(Nfield)
    
    plt.sca(ax[1][0])
    plt.errorbar(phi1, phi2, color='k', fmt='o', label='HST')
    
    plt.sca(ax[1][1])
    plt.errorbar(phi1, pmra, yerr=pmerr, color='k', fmt='o')
    
    plt.sca(ax[1][2])
    plt.errorbar(phi1, pmdec, yerr=pmerr, color='k', fmt='o')
    
    
    plt.sca(ax[0][0])
    plt.text(0.1,0.85, '$\\theta_{roll}$ = 60$^\circ$', fontsize='small', transform=plt.gca().transAxes)

    plt.sca(ax[1][0])
    plt.text(0.1,0.85, '$\\theta_{roll}$ = -45$^\circ$', fontsize='small', transform=plt.gca().transAxes)
    plt.legend(fontsize='small', loc=3, handlelength=0.2)
    
    plt.suptitle('Expected astrometric performance', fontsize='medium')
    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig('../plots/astrometric_performance.png')

def sph2cart(ra, dec):
    """Convert two angles on a unit sphere to a 3d vector"""
    
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    
    return (x, y, z)

def obs_scaling(seed=425, th=150, param='M'):
    """Project results of an encounter in a log potential on the sky for different impact parameters"""
    
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
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
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
    #plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms)
    
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    xeqs = [xeq.ra, xeq.dec, xeq.distance.to(u.kpc)]
    for i in range(3):
        plt.sca(ax[i+1])
        
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        #plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms)
        
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    if param=='M':
        plist = np.array([1e6,5e6,1e7,5e7,1e8])*u.Msun
        plist = np.array([4e7,6e7,8e7,1e8])*u.Msun
    elif param=='B':
        plist = np.array([19.55, 19.65, 19.75, 19.85])*u.kpc
    elif param=='th':
        plist = np.array([90, 110, 130, 150])*u.deg
    elif param=='V':
        plist = np.array([175, 190, 205, 220])*u.km/u.s
    elif param=='T':
        plist = np.array([0.425, 0.45,0.475, 0.5])*u.Gyr
        
    for e, p in enumerate(plist[:]):
        if param=='M':
            M = p
        elif param=='B':
            B = p
        elif param=='th':
            theta = coord.Angle(p)
        elif param=='V':
            V = p
        elif param=='T':
            T = p
        
        # scaled stream
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(plist)) + 0.35)
        ms = 2.5*(e+1)
        zorder = np.size(plist)-e
        label = '{} = {:g}'.format(param, p)
        print(e, p, color)
        
        plt.sca(ax[0])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label)
        
        for i in range(3):
            plt.sca(ax[i+1])
            vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
            plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms, zorder=zorder)
    
    plt.sca(ax[0])
    plt.legend(loc=2,fontsize='small')
    
    plt.tight_layout()
    plt.savefig('../plots/spur_observable_scaling_{}.pdf'.format(param))
    plt.savefig('../plots/spur_observable_scaling_{}.png'.format(param))


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


#############
# HST figures

def generate_fiducial(seed=235, th=150, time=0.5):
    """Generate fiducial stream in a log halo that qualitatively reproduces features observed in GD-1"""
    
    # impact parameters
    M = 3e7*u.Msun
    B = 19.95*u.kpc
    B = 20.06*u.kpc
    V = 190*u.km/u.s
    phi = coord.Angle(0*u.deg)
    th = 155
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = time*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 4000
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    observer = {'z_sun': -2000.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 50*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    #xphi0 = np.linspace(-0.1*np.pi, 0.1*np.pi, 2000)
    #xphi1 = np.linspace(-0.28*np.pi, -0.1*np.pi, 400)
    #xphi2 = np.linspace(0.1*np.pi, 0.32*np.pi, 400)
    #xphi = np.concatenate([xphi1, xphi0, xphi2])
    
    xr = 20*u.kpc + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    # closest to impact
    ienc = np.argmin(np.abs(x))
    
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
    R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    # place gap at xi~0
    xioff = xi0[ienc]
    xi0 -= xioff
    
    xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
    xi = coord.Angle(xi*u.deg)
    xi -= xioff
    
    # velocity differences
    dv = []
    for i in range(3):
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        dv += [veq[i]-vexp]
    
    outdict = {'x': stream['x'], 'v': stream['v'], 'xi': xi, 'eta': eta, 'veq': veq, 'dv': dv, 'observer': observer, 'vobs': vobs, 'R': R, 'xioff': xioff, 'x0': stream0['x'], 'v0': stream0['v'], 'xi0': xi0, 'eta0': eta, 'veq0': veq0}
    pickle.dump(outdict, open('../data/gd1_fiducial_t{:.4f}.pkl'.format(time), 'wb'))

def fiducial_snapshots(Nsnap=5):
    """Generate a Nsnap snapshots of the fiducial GD-1 model"""
    
    times = np.linspace(0,0.5,Nsnap)
    
    for t in times:
        generate_fiducial(time=t)


def fiducial_evolution():
    """Show fiducial model evolving"""
    
    # fiducial model
    wangle = 180*u.deg
    pk = pickle.load(open('../data/gd1_fiducial.pkl', 'rb'))
    x = pk['x'].to(u.kpc)
    xorig = x[:2]
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    plt.sca(ax)
    
    Nsnap = 8
    times = np.linspace(0,0.5,Nsnap)[::-1]
    angles = np.linspace(0,322,Nsnap)[::-1]*u.deg

    for e, t in enumerate(times):
        c = mpl.cm.Blues(0.05+0.85*(Nsnap-e)/Nsnap)
        #a = 0.5 + 0.5*(Nsnap-e)/Nsnap
        
        pk = pickle.load(open('../data/gd1_fiducial_t{:.4f}.pkl'.format(t), 'rb'))
        x = pk['x'].to(u.kpc)
        x_, y_ = x[0], x[1]
        
        plt.plot(x_[120:-120], y_[120:-120], '.', color=c, ms=10, zorder=Nsnap-e, rasterized=False)
        
        xt = 24*np.cos(angles[e]+90*u.deg)
        yt = 24*np.sin(angles[e]+90*u.deg)
        if e<Nsnap-1:
            txt = plt.text(xt, yt, '+ {:.2f} Gyr'.format(t), va='center', ha='center', fontsize='small', color='0.2', rotation=(angles[e]).value, zorder=10)
            txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
    
    plt.text(0, 24, 'Flyby', va='center', ha='center', fontsize='small', color='0.2')

    lim = 27
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.gca().set_aspect('equal')
    
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/loop_evolution.pdf')

def fiducial_comparison():
    """Compare fiducial model to the observed GD-1"""
    
    # observed gd1
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    
    # fiducial model
    wangle = 180*u.deg
    pk = pickle.load(open('../data/gd1_fiducial.pkl', 'rb'))
    xi, eta = pk['xi'], pk['eta']
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(7,6), sharex=True)
    
    plt.sca(ax[0])
    plt.scatter(g['phi1']+40, g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, rasterized=True)
    
    plt.ylabel('$\phi_2$ [deg]')
    plt.text(0.05, 0.9, 'Most likely GD-1 members', transform=plt.gca().transAxes, va='top', fontsize=17)
    plt.xlim(-20,20)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[1])
    plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=mpl.cm.Blues(0.9), ms=5)
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    plt.text(0.05, 0.9, 'Simulated GD-1\n0.5 Gyr after subhalo flyby', transform=plt.gca().transAxes, va='top', fontsize=17)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../plots/fiducial_comparison.pdf')

def fiducial_convolution():
    """"""
    # fiducial model
    wangle = 180*u.deg
    pk = pickle.load(open('../data/gd1_fiducial_t0.5500.pkl', 'rb'))
    xi, eta, veq, dv = pk['xi'], pk['eta']*u.deg, pk['veq'], pk['dv']
    Nstar = np.size(eta)
    
    spur_mask = (eta>0.5*u.deg) & (xi>5*u.deg) & (xi<10*u.deg)
    stream_mask = (eta<0.5*u.deg) & (xi>5*u.deg) & (xi<10*u.deg)
    altstream_mask = (eta<0.5*u.deg) & (xi<-5*u.deg) & (xi>-10*u.deg)
    masks = [spur_mask, stream_mask, altstream_mask]
    
    rasterized = True
    c = '0.35'
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(10,10), sharex=True) #, gridspec_kw = {'width_ratios':[2.5, 1]})
    
    plt.sca(ax[0])
    plt.plot(xi.wrap_at(wangle), eta, '.', color=c, label='Fiducial model', rasterized=rasterized)
    
    plt.gca().set_aspect('equal')
    plt.xlim(-30,30)
    plt.ylim(-10,10)
    plt.ylabel('$\phi_2$ [deg]')
    txt = plt.text(0.97,0.83, '(a)', transform=plt.gca().transAxes, ha='right', fontsize=17)
    
    plt.sca(ax[1])
    plt.plot(xi.wrap_at(wangle), dv[0], '.', color=c, zorder=2, rasterized=rasterized)
    
    txt = plt.text(0.03, 0.8, 'Fiducial model', transform=plt.gca().transAxes)
    plt.ylim(-1,1)
    plt.ylabel('$\Delta$ $\mu_{\\alpha_\star}$ [mas yr$^{-1}$]')
    txt = plt.text(0.97,0.83, '(b)', transform=plt.gca().transAxes, ha='right', fontsize=17)
    
    plt.sca(ax[2])
    #0.37
    plt.plot(xi.wrap_at(wangle), dv[0] + np.random.randn(Nstar)*0.25*u.mas/u.yr, '.', color=c, zorder=0, rasterized=rasterized)

    txt = plt.text(0.03, 0.8, 'Gaia forecast', transform=plt.gca().transAxes)
    txt.set_bbox(dict(facecolor='w', alpha=0.85, ec='none'))
    plt.ylim(-1,1)
    plt.ylabel('$\Delta$ $\mu_{\\alpha_\star}$ [mas yr$^{-1}$]')
    txt = plt.text(0.97,0.83, '(c)', transform=plt.gca().transAxes, ha='right', fontsize=17)
    txt.set_bbox(dict(facecolor='w', alpha=0.85, ec='none'))

    plt.sca(ax[3])
    loop = np.abs(dv[0])>0.1*u.mas/u.yr
    #plt.plot(xi.wrap_at(wangle), dv[0] + np.random.randn(Nstar)*0.06*u.mas/u.yr, '.', color='0.4', zorder=1)
    plt.plot(xi[~loop].wrap_at(wangle), dv[0][~loop] + np.random.randn(np.sum(~loop))*0.06*u.mas/u.yr, '.', color=c, zorder=1, rasterized=rasterized)
    plt.plot(xi[loop].wrap_at(wangle), dv[0][loop] + np.random.randn(np.sum(loop))*0.08*u.mas/u.yr, '.', color=c, zorder=1, rasterized=rasterized)
    
    txt = plt.text(0.03, 0.8, 'HST forecast', transform=plt.gca().transAxes)
    plt.ylim(-1,1)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $\mu_{\\alpha_\star}$ [mas yr$^{-1}$]')
    txt = plt.text(0.97,0.83, '(d)', transform=plt.gca().transAxes, ha='right', fontsize=17)
    #for m in masks:
        #plt.plot(xi.wrap_at(wangle)[m], dv[0][m], '.')
    
    #plt.gca().set_aspect('equal')
    #plt.ylim(0,1)
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/pm_forecasts.pdf', dpi=150)

def fiducial_convolution2(seed=425, th=150, alt=False):
    """Project results of an encounter in a log potential on the sky"""
    
    # impact parameters
    M = 3e7*u.Msun
    B = 19.95*u.kpc
    B = 20.08*u.kpc
    V = 190*u.km/u.s
    phi = coord.Angle(0*u.deg)
    th = 155
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    old_label = ''
    np.random.seed(seed)
    observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    if alt:
        label = '_alt'
        observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
        vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0,0,0]*u.km/u.s}
        
        # impact parameters
        M = 3e7*u.Msun
        B = 20.08*u.kpc
        V = 190*u.km/u.s
        phi = coord.Angle(0*u.deg)
        th = 155
        theta = coord.Angle(th*u.deg)
        Tenc = 0.01*u.Gyr
        T = 0.5*u.Gyr
        dt = 0.05*u.Myr
        #dt = 1*u.Myr
        rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 1400
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xphi0 = np.linspace(-0.1*np.pi, 0.1*np.pi, 1000)
    xphi1 = np.linspace(-0.28*np.pi, -0.1*np.pi, 200)
    xphi2 = np.linspace(0.1*np.pi, 0.32*np.pi, 200)
    xphi = np.concatenate([xphi1, xphi0, xphi2])
    
    xr = 20*u.kpc + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh# * 0.94
    vy = np.sin(xphi) * Vh #* 0.97
    vz = vx * 0
    # closest to impact
    ienc = np.argmin(np.abs(x))
    
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
    R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    # place gap at xi~0
    xioff = xi0[ienc]
    xi0 -= xioff
    
    xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
    xi = coord.Angle(xi*u.deg)
    xi -= xioff
    
    vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    ylims = [[-0.5, 0.5], [-0.5, 0.5], [-25,25]]
    color = '0.35'
    ms = 4
    
    # plotting
    plt.close()
    fig, ax = plt.subplots(5,1,figsize=(12,12), sharex=True)
    
    plt.sca(ax[0])
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    plt.scatter(g['phi1']+40, g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1)
    
    plt.xlim(-45,45)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    plt.ylabel('$\phi_1$ [deg]')
    
    plt.sca(ax[1])
    plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms)
    
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    xeqs = [xeq.ra, xeq.dec, xeq.distance.to(u.kpc)]
    for i in range(3):
        plt.sca(ax[i+2])
        
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms)
        
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    plt.tight_layout()
    #plt.savefig('../plots/{}.png'.format(label))


def pm_precision(seed=425, th=150):
    """Compare gaia & HST proper motions for the stream & the spur"""
    
    # observed gd1
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    
    # fiducial model
    wangle = 180*u.deg
    pk = pickle.load(open('../data/gd1_fiducial.pkl', 'rb'))
    xi, eta, veq, xi0, eta0, veq0 = pk['xi'], pk['eta'], pk['veq'], pk['xi0'], pk['eta0'], pk['veq0']
    
    # velocity differences
    dv = []
    for i in range(3):
        # interpolate expected kinematics from an unperturbed stream
        vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
        dv += [veq[i]-vexp]
    
    # find closest model star to the gd-1 stars
    Ngd1 = len(g)
    p = np.array([g['phi1']+40, g['phi2']])
    q = np.array([xi.wrap_at(wangle).to(u.deg).value, eta])
    idmin = np.empty(Ngd1, dtype='int')
    
    for i in range(Ngd1):
        dist = np.sqrt((p[0,i]-q[0])**2 + (p[1,i]-q[1])**2)
        idmin[i] = np.argmin(dist)

    # mask stream, mask spur
    onstream_mask = ((g['phi1']<-30.5) & (g['phi1']>-35.5) & (g['phi2']>-0.2) & (g['phi2']<0.2))
    spur_mask = ((g['phi1']<-30.5) & (g['phi1']>-35.5) & (g['phi2']>1) & (g['phi2']<1.4))
    #onstream_mask = ((g['phi1']<-27) & (g['phi1']>-35.5) & (g['phi2']>-0.2) & (g['phi2']<0.2))
    #spur_mask = ((g['phi1']<-27) & (g['phi1']>-35.5) & (g['phi2']>1) & (g['phi2']<1.4))
    all_mask = np.ones(Ngd1, dtype='bool')
    
    # plot scaled data uncertainties on model pm drawn from a corresponding obs uncertainty
    np.random.seed(seed+1)
    fgaia = np.sqrt(2/5)
    pmra_error = fgaia * g['pmra_error']*u.mas/u.yr
    pmdec_error = fgaia * g['pmdec_error']*u.mas/u.yr
    print(np.median(pmra_error), np.median(pmdec_error))

    phi1 = xi[idmin].wrap_at(wangle).to(u.deg).value
    phi2 = eta[idmin]
    pmra = dv[0][idmin] + pmra_error * np.random.randn(Ngd1)
    pmdec = dv[1][idmin] + pmdec_error * np.random.randn(Ngd1)
    pmra_true = dv[0][idmin]
    pmdec_true = dv[1][idmin]
    
    # convolve HST uncertainties
    pmerr = np.array([0.0848, 0.0685])
    Nfield = 2
    #p2 = np.array([np.array([-32.77,-32.77])+40, [1.167,0]])
    p2 = np.array([np.array([-27, -27])+40, [1.167,0]])
    q = np.array([xi.wrap_at(wangle).to(u.deg).value, eta])
    idmin2 = np.empty(Nfield, dtype='int')
    
    for i in range(Nfield):
        dist = np.sqrt((p2[0,i]-q[0])**2 + (p2[1,i]-q[1])**2)
        idmin2[i] = np.argmin(dist)
    
    np.random.seed(seed+7)
    phi1 = xi[idmin2].wrap_at(wangle).to(u.deg).value
    phi2 = eta[idmin2]
    pmra2 = dv[0][idmin2].value + pmerr * np.random.randn(Nfield)
    pmdec2 = dv[1][idmin2].value + pmerr * np.random.randn(Nfield)
    pmra2_true = dv[0][idmin2].value
    pmdec2_true = dv[1][idmin2].value
    
    # velocity scaling
    dist = gd1_dist(coord.Angle(-32.77*u.deg)).to(u.kpc).value
    r_v = np.array([5,10,20,40,70])
    r_pm = r_v/(4.74*dist)
    
    # mass scaling
    verr = pmerr*4.74*dist*u.km/u.s
    vkick = (G*3e7*u.Msun/(50*u.pc*190*u.km/u.s*np.sin(90*u.deg))).to(u.km/u.s)
    dM = (3*verr*50*u.pc*190*u.km/u.s/G).to(u.Msun)
    dMM = 3*verr/vkick
    #print(vkick)
    #print('{:g} {:g}'.format(*dMM))
    
    
    cspur = mpl.cm.magma(0.6)
    cstream = mpl.cm.magma(0.2)
    colors = [cspur, cstream]
    alpha = [0.1, 0.07]
    lw = [2.5, 3.5]
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10.6,5.3))
    
    for i in range(2):
        plt.sca(ax[i])
        plt.errorbar(pmra[onstream_mask].value, pmdec[onstream_mask].value, yerr=pmdec_error[onstream_mask].value, xerr=pmra_error[onstream_mask].value, fmt='none', color=cstream, alpha=alpha[i], lw=lw[i])
        plt.errorbar(pmra[spur_mask].value, pmdec[spur_mask].value, yerr=pmdec_error[spur_mask].value, xerr=pmra_error[spur_mask].value, fmt='none', color=cspur, alpha=alpha[i], lw=lw[i])
        
        for e in range(2):
            plt.plot(pmra2_true[e], pmdec2_true[e], 'x', color=colors[e], ms=14, mew=4)
            plt.errorbar(pmra2[e], pmdec2[e], yerr=pmerr[e], xerr=pmerr[e], fmt='o', color=colors[e], lw=2.5)
        
        # add absolute velocity contours
        for r in r_pm:
            c = mpl.patches.Circle((0,0), radius=r, fc='none', ec='k', lw=1.5, ls=':', alpha=0.5)
            plt.gca().add_patch(c)
    
        plt.xlabel('$\Delta$ $\mu_{\\alpha_\star}$ [mas yr$^{-1}$]')
        plt.ylabel('$\Delta$ $\mu_\delta$ [mas yr$^{-1}$]')
    
    plt.sca(ax[0])
    # legend entries
    plt.errorbar(20+pmra[onstream_mask].value, pmdec[onstream_mask].value, yerr=pmdec_error[onstream_mask].value, xerr=pmra_error[onstream_mask].value, fmt='none', color='k', alpha=0.3, lw=lw[0], label='Gaia DR4', zorder=0)
    plt.errorbar(20+pmra2, pmdec2, yerr=pmerr, xerr=pmerr, fmt='o', color='0.3', lw=2.5, label='HST', zorder=1)
    plt.plot(20+pmra2_true, pmdec2_true, 'x', color='0.3', ms=10, mew=4, label='GD-1 model', zorder=2)
    plt.plot(20+pmra2_true, pmdec2_true, 's', color=cstream, ms=11, alpha=0.8, mew=0, label='Stream', zorder=3)
    plt.plot(20+pmra2_true, pmdec2_true, 's', color=cspur, ms=11, alpha=0.8, mew=0, label='Spur', zorder=4)
    
    plt.legend(fontsize='small', loc=2, ncol=2, handlelength=0.9)
    
    # resort the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.gca().legend(handles, labels, fontsize='small', loc=2, ncol=2, handlelength=0.9)

    # velocity labels
    plt.text(0.1, -0.75, '20 km s$^{-1}$', fontsize='small', ha='center', color='0.2')
    plt.text(0.37, -1.13, '40 km s$^{-1}$', fontsize='small', ha='left', color='0.2')
    plt.text(1, -1.66, '70 km s$^{-1}$', fontsize='small', ha='left', color='0.2')
    
    # zoom in guidelines
    r = mpl.patches.Rectangle((-0.5,-0.5), 1, 1, fc='none', ec='k')
    plt.gca().add_patch(r)
    ax[0].annotate('', xy=(0.5,-0.5), xytext=(0,0), xycoords=ax[0].transData, textcoords=ax[1].transAxes, arrowprops=dict(color='k', arrowstyle='-'))
    ax[0].annotate('', xy=(0.5,0.5), xytext=(0,1), xycoords=ax[0].transData, textcoords=ax[1].transAxes, arrowprops=dict(color='k', arrowstyle='-'))
    
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[1])
    # velocity labels
    plt.text(0.1, -0.18, '5 km s$^{-1}$', fontsize='small', ha='right', color='0.2')
    plt.text(0.1, -0.31, '10 km s$^{-1}$', fontsize='small', ha='right', color='0.2')
    plt.text(0.1, -0.46, '20 km s$^{-1}$', fontsize='small', ha='right', color='0.2')
    
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,0.5)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../plots/pm_precision.pdf')


def gd1_dist(phi1):
    # 0, 10
    # -60, 7
    m = (10-7) / (60)
    return (m*phi1.wrap_at(180*u.deg).value + 10) * u.kpc


###########
# Scalings

def generate_variations(seed=425, th=150):
    """Generate streams at a range of varied parameters"""
    
    # impact parameters
    M = 1e8*u.Msun
    B0 = 19.85*u.kpc

    V = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    potential_perturb = 1
    
    # potential parameters (log halo)
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 1400
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xphi0 = np.linspace(-0.1*np.pi, 0.1*np.pi, 1000)
    xphi1 = np.linspace(-0.28*np.pi, -0.1*np.pi, 200)
    xphi2 = np.linspace(0.1*np.pi, 0.32*np.pi, 200)
    xphi = np.concatenate([xphi1, xphi0, xphi2])
    
    Bs = 20*u.kpc
    xr = Bs + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    ienc = np.argmin(np.abs(x))
    
    farray = np.array([0.1, 0.3, 0.5, 1, 2, 3, 10])
    farray = np.array([0.3,0.5,0.8,0.9,1,1.1,1.2,2,3])
    
    for e, f in enumerate(farray):
        # unperturbed stream
        par_perturb = np.array([0*M.si.value, 0., 0., 0.])
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T*f).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
        xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
        xi0 = coord.Angle(xi0*u.deg)
        
        # place gap at xi~0
        xioff = xi0[ienc]
        xi0 -= xioff
        
        par_perturb = np.array([M.si.value, 0., 0., 0.])
        dB = (B0 - Bs)
        B = dB + Bs
        
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T*f).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        outdict = {'x': stream['x'], 'v': stream['v'], 'xi': xi, 'eta': eta, 'observer': observer, 'vobs': vobs, 'R': R, 'xi0': xioff, 'x0': stream0['x'], 'v0': stream0['v']}
        pickle.dump(outdict, open('../data/variations/vary_th{:03d}_T_{:.1f}.pkl'.format(th, f), 'wb'))
    
    # unperturbed stream
    par_perturb = np.array([0*M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    # place gap at xi~0
    xioff = xi0[ienc]
    xi0 -= xioff
    
    for e, f in enumerate(farray):
        par_perturb = np.array([f*M.si.value, 0., 0., 0.])
        dB = (B0 - Bs)
        B = dB + Bs
    
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        outdict = {'x': stream['x'], 'v': stream['v'], 'xi': xi, 'eta': eta, 'observer': observer, 'vobs': vobs, 'R': R, 'xi0': xioff, 'x0': stream0['x'], 'v0': stream0['v']}
        pickle.dump(outdict, open('../data/variations/vary_th{:03d}_M_{:.1f}.pkl'.format(th, f), 'wb'))
    
    for e, f in enumerate(farray):
        par_perturb = np.array([M.si.value, 0., 0., 0.])
        dB = (B0 - Bs)
        B = dB*f + Bs
        
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        outdict = {'x': stream['x'], 'v': stream['v'], 'xi': xi, 'eta': eta, 'observer': observer, 'vobs': vobs, 'R': R, 'xi0': xioff, 'x0': stream0['x'], 'v0': stream0['v']}
        pickle.dump(outdict, open('../data/variations/vary_th{:03d}_B_{:.1f}.pkl'.format(th, f), 'wb'))
    
    theta0 = theta
    V0 = V
    
    for e, f in enumerate(farray):
        par_perturb = np.array([M.si.value, 0., 0., 0.])
        dB = (B0 - Bs)
        B = dB + Bs
        
        vpar = Vh + np.cos(theta0.rad)*V0
        vperp = np.sin(theta0.rad)*V0
        vpar_scaled = vpar*f
        vperp_scaled = vperp*f
        
        V = np.sqrt((vpar_scaled-Vh)**2 + vperp_scaled**2)
        theta = coord.Angle(np.arctan2(vperp_scaled, vpar_scaled-Vh))
        
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        outdict = {'x': stream['x'], 'v': stream['v'], 'xi': xi, 'eta': eta, 'observer': observer, 'vobs': vobs, 'R': R, 'xi0': xioff, 'x0': stream0['x'], 'v0': stream0['v']}
        pickle.dump(outdict, open('../data/variations/vary_th{:03d}_V_{:.1f}.pkl'.format(th, f), 'wb'))

def plot_variations(th=150):
    """"""
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(14,8), sharex=True, sharey=True)
    
    farray = np.array([0.8, 1, 1.2])
    #farray = np.array([0.3,0.5, 1, 2,3])
    Nf = np.size(farray)
    labels = ['M', 'T', 'B', 'V']
    cmaps = [mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    lg_scale = ['0.5 ', '', '2 ']
    lg_scale = ['0.8 ', '', '1.2 ']
    #lg_scale = ['0.3 ', '0.5 ', '', '2 ', '3 ']
    lg_text = ['M', 'T', 'B', 'V$_\perp$']
    
    for i in range(4):
        irow = np.int(i/2)
        icol = i%2
        plt.sca(ax[irow][icol])
        
        for e, f in enumerate(farray):
            var = pickle.load(open('../data/variations/vary_th{:03d}_{:s}_{:.1f}.pkl'.format(th, labels[i], f), 'rb'))
            label = '{}{}'.format(lg_scale[e], lg_text[i])
            
            if irow==0:
                zorder = Nf - e
                fcolor = (1 + e)/(Nf+1)
            else:
                zorder = e
                fcolor = (Nf - e)/(Nf+1)
            
            plt.plot(var['xi'].wrap_at(180*u.deg), var['eta'], 'o', mec='none', color=cmaps[i](fcolor), label=label, zorder=zorder)
        
        plt.legend(fontsize='small', handlelength=0.2)
    
    plt.xlim(-45,45)
    plt.ylim(-5,10)
    #plt.gca().set_aspect('equal')
    
    plt.sca(ax[0][0])
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1][0])
    plt.ylabel('$\phi_2$ [deg]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.sca(ax[1][1])
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/variations_diff_th{:03d}.png'.format(th))

def plot_vvariations(th=150, vind=0):
    """"""
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(14,8), sharex=True, sharey=True)
    
    farray = np.array([0.5, 1, 2])
    farray = np.array([0.8, 1, 1.2])
    #farray = np.array([0.3,0.5, 1, 2,3])
    Nf = np.size(farray)
    labels = ['M', 'T', 'B', 'V']
    cmaps = [mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    lg_scale = ['0.5 ', '', '2 ']
    lg_scale = ['0.8 ', '', '1.2 ']
    #lg_scale = ['0.3 ', '0.5 ', '', '2 ', '3 ']
    lg_text = ['M', 'T', 'B', 'V$_\perp$']
    
    for i in range(4):
        irow = np.int(i/2)
        icol = i%2
        plt.sca(ax[irow][icol])
        
        for e, f in enumerate(farray):
            var = pickle.load(open('../data/variations/vary_th{:03d}_{:s}_{:.1f}.pkl'.format(th, labels[i], f), 'rb'))
            label = '{}{}'.format(lg_scale[e], lg_text[i])
            
            if irow==0:
                zorder = Nf - e
                fcolor = (1 + e)/(Nf+1)
            else:
                zorder = e
                fcolor = (Nf - e)/(Nf+1)
            
            # sky coordinates
            xgal = coord.Galactocentric(var['x'], **var['observer'])
            xeq = xgal.transform_to(coord.ICRS)
            veq_ = gc.vgal_to_hel(xeq, var['v'], **var['vobs'])
            veq = [None] * 3
            veq[0] = veq_[0].to(u.mas/u.yr)
            veq[1] = veq_[1].to(u.mas/u.yr)
            veq[2] = veq_[2].to(u.km/u.s)
            
            # unperturbed stream
            xgal0 = coord.Galactocentric(var['x0'], **var['observer'])
            xeq0 = xgal0.transform_to(coord.ICRS)
            veq0_ = gc.vgal_to_hel(xeq0, var['v0'], **var['vobs'])
            veq0 = [None] * 3
            veq0[0] = veq0_[0].to(u.mas/u.yr)
            veq0[1] = veq0_[1].to(u.mas/u.yr)
            veq0[2] = veq0_[2].to(u.km/u.s)
            
            xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, var['R'])
            xi0 = coord.Angle(xi0*u.deg) - var['xi0']
            
            # interpolate expected kinematics from an unperturbed stream
            wangle = 180*u.deg
            vexp = np.interp(var['xi'].wrap_at(wangle), xi0.wrap_at(wangle), veq0[vind].value) * veq0[vind].unit
            plt.plot(var['xi'].wrap_at(wangle), veq[vind]-vexp, 'o', mec='none', color=cmaps[i](fcolor), label=label, zorder=zorder)
            
            #plt.plot(var['xi'].wrap_at(180*u.deg), veq[vind], 'o', mec='none', color=cmaps[i](fcolor), label=label, zorder=zorder)
        
        plt.legend(fontsize='small', handlelength=0.2)
    
    ylims = [[-0.5,0.5],[-0.5,0.5],[-30,20]]
    ylabels = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    
    plt.xlim(-45,45)
    plt.ylim(ylims[vind][0],ylims[vind][1])
    
    plt.sca(ax[0][0])
    plt.ylabel('$\Delta$ {}'.format(ylabels[vind]))
    
    plt.sca(ax[1][0])
    plt.ylabel('$\Delta$ {}'.format(ylabels[vind]))
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.sca(ax[1][1])
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/variations_diff_th{:03d}_v{:1d}.png'.format(th, vind))

def compare_effects(pairs=[0,3], th=150, diff=True):
    """"""
    
    plt.close()
    fig, ax = plt.subplots(1,4,figsize=(16,4), sharex=True)
    
    wangle = 180*u.deg
    if diff:
        farray = np.array([0.9, 1, 1.1])
        lg_scale = ['0.9 ', '', '1.1 ']
        diff_label = '_diff'
    else:
        farray = np.array([0.8, 1, 1.2])
        lg_scale = ['0.8 ', '', '1.2 ']
        diff_label = ''
    Nf = np.size(farray)
    labels = ['M', 'T', 'B', 'V']
    lg_text = ['M', 'T', 'B', 'V$_\perp$']
    cmaps = [mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    
    for i in pairs:
        for e, f in enumerate(farray):
            var = pickle.load(open('../data/variations/vary_th{:03d}_{:s}_{:.1f}.pkl'.format(th, labels[i], f), 'rb'))
            label = '{}{}'.format(lg_scale[e], lg_text[i])
            
            if i==0:
                zorder = Nf - e
                fcolor = (1 + e)/(Nf+1)
            else:
                zorder = e
                fcolor = (Nf - e)/(Nf+1)
            
            # sky coordinates
            xgal = coord.Galactocentric(var['x'], **var['observer'])
            xeq = xgal.transform_to(coord.ICRS)
            veq_ = gc.vgal_to_hel(xeq, var['v'], **var['vobs'])
            veq = [None] * 3
            veq[0] = veq_[0].to(u.mas/u.yr)
            veq[1] = veq_[1].to(u.mas/u.yr)
            veq[2] = veq_[2].to(u.km/u.s)
            
            # unperturbed stream
            xgal0 = coord.Galactocentric(var['x0'], **var['observer'])
            xeq0 = xgal0.transform_to(coord.ICRS)
            veq0_ = gc.vgal_to_hel(xeq0, var['v0'], **var['vobs'])
            veq0 = [None] * 3
            veq0[0] = veq0_[0].to(u.mas/u.yr)
            veq0[1] = veq0_[1].to(u.mas/u.yr)
            veq0[2] = veq0_[2].to(u.km/u.s)
            
            xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, var['R'])
            xi0 = coord.Angle(xi0*u.deg) - var['xi0']
            
            plt.sca(ax[0])
            plt.plot(var['xi'].wrap_at(180*u.deg), var['eta'], 'o', mec='none', color=cmaps[i](fcolor), label=label, zorder=zorder)
    
            for vind in range(3):
                plt.sca(ax[vind+1])
                
                # interpolate expected kinematics from an unperturbed stream
                vexp = np.interp(var['xi'].wrap_at(wangle), xi0.wrap_at(wangle), veq0[vind].value) * veq0[vind].unit
                plt.plot(var['xi'].wrap_at(wangle), veq[vind]-vexp, 'o', mec='none', color=cmaps[i](fcolor), label=label, zorder=zorder)
            
            
    ylims = [[-2,5], [-0.5,0.5],[-0.5,0.5],[-30,20]]
    ylabels = ['$\phi_2$ [deg]', '$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']

    for i in range(4):
        plt.sca(ax[i])
        plt.xlim(-45,45)
        plt.ylim(ylims[i][0],ylims[i][1])
        
        plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('$\Delta$ {}'.format(ylabels[i]))
    
    plt.sca(ax[0])
    plt.legend(fontsize='x-small', ncol=2, handlelength=0.2)
    
    plt.tight_layout()
    plt.savefig('../plots/effects{}_{}{}.png'.format(diff_label, labels[pairs[0]], labels[pairs[1]]))


def vary_time(seed=425, th=150, fmass=1, fb=1, rfig=False):
    """"""
    
    # impact parameters
    M = 1e8*u.Msun
    B0 = 19.85*u.kpc
    V = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 220*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 20*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # setup tube
    Nstar = 1400
    wx = 30*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 60*u.deg, 'galcen_coord': coord.SkyCoord(ra=300*u.deg, dec=-90*u.deg, frame='icrs')}
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    xphi0 = np.linspace(-0.1*np.pi, 0.1*np.pi, 1000)
    xphi1 = np.linspace(-0.28*np.pi, -0.1*np.pi, 200)
    xphi2 = np.linspace(0.1*np.pi, 0.32*np.pi, 200)
    xphi = np.concatenate([xphi1, xphi0, xphi2])
    
    Bs = 20*u.kpc
    xr = Bs + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    ienc = np.argmin(np.abs(x))
    
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    
    farray = np.array([0.5, 1, 2])
    
    rasterized = False
    if rfig:
        rasterized = True
    
    alpha = 1
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(12,12), sharex=True, sharey=True)
    
    for e, f in enumerate(farray):
        # unperturbed stream
        par_perturb = np.array([0*M.si.value, 0., 0., 0.])
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T*f).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
        xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
        xi0 = coord.Angle(xi0*u.deg)
        
        # place gap at xi~0
        xioff = xi0[ienc]
        xi0 -= xioff
        
        fsqrt = np.sqrt(f)
        par_perturb = np.array([fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB + Bs
        
        print((G*M*fmass/(np.abs(dB)*(V*np.sin(theta.rad))**2)).decompose())
        
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB*f/V).to(u.Myr)
        #print(fi)
        
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T*f).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[0])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized, alpha=alpha)
    
    # unperturbed stream
    par_perturb = np.array([0*M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    R = find_greatcircle(xeq0.ra.deg[::10], xeq0.dec.deg[::10])
    xi0, eta0 = myutils.rotate_angles(xeq0.ra, xeq0.dec, R)
    xi0 = coord.Angle(xi0*u.deg)
    
    # place gap at xi~0
    xioff = xi0[ienc]
    xi0 -= xioff
    
    for e, f in enumerate(farray):
        fsqrt = np.sqrt(f)
        par_perturb = np.array([f*fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB + Bs
        
        print((G*M*fmass*f/(np.abs(dB)*(V*np.sin(theta.rad))**2)).decompose())
        
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB*f/V).to(u.Myr)
        #print(fi)
        
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[1])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized, alpha=alpha)
    
    for e, f in enumerate(farray):
        fsqrt = np.sqrt(f)
        par_perturb = np.array([fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB*f + Bs
        
        print((G*M*fmass/(np.abs(f*dB)*(V*np.sin(theta.rad))**2)).decompose())
        
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB*f/V).to(u.Myr)
        #print(fi)
        
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[2])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized, alpha=alpha)
    
    theta0 = theta
    V0 = V
    
    for e, f in enumerate(farray):
        fsqrt = np.sqrt(f)
        par_perturb = np.array([fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB + Bs
        
        vpar = Vh + np.cos(theta0.rad)*V0
        vperp = np.sin(theta0.rad)*V0
        
        vpar_scaled = vpar
        vperp_scaled = vperp*f
        
        V = np.sqrt((vpar_scaled-Vh)**2 + vperp_scaled**2)
        theta = coord.Angle(np.arctan2(vperp_scaled, vpar_scaled-Vh))
        
        print((G*M*fmass/(np.abs(dB)*(V*np.sin(theta.rad))**2)).decompose())
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB/V).to(u.Myr)
        #print(fi)
        
        x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, (T).si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        xi -= xioff
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[3])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized, alpha=alpha)

    for i in range(4):
        plt.sca(ax[i])
        plt.ylabel('$\phi_1$ [deg]')

    plt.xlabel('$\phi_2$ [deg]')
    plt.xlim(-45,45)
    
    plt.tight_layout()
    


def scale_fixed_M2B(seed=425, th=150, fmass=1, fb=1, rfig=False):
    """Contrast stream morphology post encounter, while keeping the ratio of perturber's mass to its impact parameter fixed"""
    
    # impact parameters
    M = 1e8*u.Msun
    B0 = 19.85*u.kpc
    V = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
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
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    Bs = 20*u.kpc
    xr = Bs + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    
    # generate stream model
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    
    farray = np.array([0.3, 0.5, 1, 2, 3])
    
    rasterized = False
    if rfig:
        rasterized = True
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(12,12), sharex=True)
    
    for e, f in enumerate(farray):
        fsqrt = np.sqrt(f)
        par_perturb = np.array([f*fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB*f + Bs
        
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB*f/V).to(u.Myr)
        #print(fi)
        
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[0])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized)
        
        for i in range(3):
            plt.sca(ax[i+1])
            vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
            plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms, zorder=zorder, rasterized=rasterized)
    
    # label axes
    plt.sca(ax[0])
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.xlim(65,135)
    #plt.gca().set_aspect('equal')
    plt.legend(fontsize='x-small', loc=2)
    plt.title('f M, f B | M = {:g} | B = {:g} | $\\theta$ = {:.0f}'.format(fmass*M, dB.to(u.pc), theta), fontsize='medium')
    
    vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    ylims = [[-0.5, 0.5], [-0.5, 0.5], [-25,25]]
    ylims = [[-1,1], [-1,1], [-50,50]]
    for i in range(3):
        plt.sca(ax[i+1])
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    plt.tight_layout()
    
    if rfig:
        return fig
    else:
        plt.savefig('../plots/scale_MB_th{:03d}_{:.1f}_{:.1f}.png'.format(th, fmass, fb))

def vary_mass(verbose=False, th=90, seed=329):
    """"""
    
    farray = np.linspace(0.3, 3, 10)
    farray = np.logspace(np.log10(0.3), np.log10(3), 10)
    
    pp = PdfPages('../plots/scale_MB_th{:03d}_vM.pdf'.format(th))
    
    for f in farray:
        if verbose: print(f)
        fig = scale_fixed_M2B(seed=seed, th=th, fmass=f, rfig=True)
        pp.savefig(fig)
    
    pp.close()

def vary_distance(verbose=False, th=90, seed=329):
    """"""
    
    farray = np.linspace(0.3, 3, 10)
    farray = np.logspace(np.log10(0.3), np.log10(3), 10)
    
    pp = PdfPages('../plots/scale_MB_th{:03d}_vB.pdf'.format(th))
    
    for f in farray:
        if verbose: print(f)
        fig = scale_fixed_M2B(seed=seed, th=th, fb=f, rfig=True)
        pp.savefig(fig)
    
    pp.close()


def scale_fixed_M2V(seed=425, th=150, fmass=1, fb=1, fv=1, rfig=False):
    """Contrast stream morphology post encounter, while keeping the ratio of perturber's mass to its impact parameter fixed"""
    
    # impact parameters
    M = 1e8*u.Msun
    B0 = 19.85*u.kpc
    V0 = 220*u.km/u.s
    phi = coord.Angle(0*u.deg)
    theta0 = coord.Angle(th*u.deg)
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
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
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    Bs = 20*u.kpc
    xr = Bs + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    
    # generate stream model
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V0.si.value, theta0.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V0.si.value, theta0.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    
    farray = np.array([0.3, 0.5, 1, 2, 3])
    #farray = np.array([0.5, 1, 2])
    #farray = np.array([0.5, 1])
    
    rasterized = False
    if rfig:
        rasterized = True
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(12,12), sharex=True, squeeze=False)
    
    for e, f in enumerate(farray):
        fsqrt = np.sqrt(f)
        par_perturb = np.array([f*fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB + Bs
        
        vpar = Vh + np.cos(theta0.rad)*V0
        vperp = np.sin(theta0.rad)*V0
        
        vpar_scaled = vpar*f
        vperp_scaled = vperp*f
        
        V = np.sqrt((vpar_scaled-Vh)**2 + vperp_scaled**2)
        theta = coord.Angle(np.arctan2(vperp_scaled, vpar_scaled-Vh))
        
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB/(vperp_scaled)).to(u.Myr)
        #print(fi)
        
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[0][0])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized)
        
        #for i in range(3):
            #plt.sca(ax[i+1])
            #vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
            #plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms, zorder=zorder, rasterized=rasterized)
    
    # label axes
    plt.sca(ax[0][0])
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.xlim(65,135)
    #plt.gca().set_aspect('equal')
    plt.legend(fontsize='x-small', loc=2)
    plt.title('f M, f V | M = {:g} | V = {:g} | $\\theta$ = {:.0f}'.format(fmass*M, V.to(u.km/u.s), theta.to(u.deg)), fontsize='medium')
    
    #vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    #ylims = [[-0.5, 0.5], [-0.5, 0.5], [-25,25]]
    #ylims = [[-1,1], [-1,1], [-50,50]]
    #for i in range(3):
        #plt.sca(ax[i+1])
        #plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        #plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    plt.tight_layout()
    
    if rfig:
        return fig
    else:
        plt.savefig('../plots/scale_MV_th{:03d}_{:.1f}_{:.1f}.png'.format(th, fmass, fv))

def vary_speed(verbose=False, th=90, seed=329):
    """"""
    
    farray = np.linspace(0.3, 3, 10)
    farray = np.logspace(np.log10(0.3), np.log10(3), 10)
    
    pp = PdfPages('../plots/scale_MV_th{:03d}_vV.pdf'.format(th))
    
    for f in farray:
        if verbose: print(f)
        fig = scale_fixed_M2V(seed=seed, th=th, fv=f, rfig=True)
        pp.savefig(fig)
    
    pp.close()


def scale_fixed_V2B(seed=425, th=150, fmass=1, fb=1, fv=1, rfig=False):
    """Contrast stream morphology post encounter, while keeping the ratio of perturber's mass to its impact parameter fixed"""
    
    # impact parameters
    M = 1e8*u.Msun
    B0 = 19.85*u.kpc
    V0 = 220*u.km/u.s
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
    vobs = {'vcirc': 220*u.km/u.s, 'vlsr': [0, 0, 0]*u.km/u.s}
    wangle = 180*u.deg
    
    xphi = np.linspace(-0.3*np.pi,0.3*np.pi, Nstar)
    Bs = 20*u.kpc
    xr = Bs + np.random.randn(Nstar)*0.0*u.kpc
    x = np.sin(xphi) * xr
    y = np.cos(xphi) * xr
    z = x * 0
    vx = -np.cos(xphi) * Vh
    vy = np.sin(xphi) * Vh
    vz = vx * 0
    
    # generate stream model
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V0.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    x1, x2, x3, v1, v2, v3 = interact.interact(par_perturb, B0.si.value, phi.rad, V0.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
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
    
    farray = np.array([0.3, 0.5, 1, 2, 3])
    
    rasterized = False
    if rfig:
        rasterized = True
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(12,12), sharex=True)
    
    for e, f in enumerate(farray):
        fsqrt = np.sqrt(f)
        par_perturb = np.array([fmass*M.si.value, 0., 0., 0.])
        #B = B0
        
        dB = (B0 - Bs)*fb
        B = dB*f + Bs
        
        V = fv*V0/fsqrt
        
        #fi = np.abs(V*T/(dB/f)).decompose()
        fi = np.abs(dB/V).to(u.Myr)
        #print(fi)
        
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
        xi, eta = myutils.rotate_angles(xeq.ra, xeq.dec, R)
        xi = coord.Angle(xi*u.deg)
        
        color = '{:f}'.format(0.65 - 0.65*(e+1)/(np.size(farray)) + 0.35)
        ms = 1.5*(e+2)
        zorder = np.size(farray)-e
        label = 'f={:g}, $t_{{imp}}$={:.1f}'.format(f, fi)
        #print(e, p, color)
        
        plt.sca(ax[0])
        plt.plot(xi.wrap_at(wangle), eta, 'o', mec='none', color=color, ms=ms, zorder=zorder, label=label, rasterized=rasterized)
        
        for i in range(3):
            plt.sca(ax[i+1])
            vexp = np.interp(xi.wrap_at(wangle), xi0.wrap_at(wangle), veq0[i].value) * veq0[i].unit
            plt.plot(xi.wrap_at(wangle), veq[i]-vexp, 'o', mec='none', color=color, ms=ms, zorder=zorder, rasterized=rasterized)
    
    # label axes
    plt.sca(ax[0])
    plt.ylabel('$\phi_1$ [deg]')
    plt.ylim(-10,10)
    plt.xlim(65,135)
    #plt.gca().set_aspect('equal')
    plt.legend(fontsize='x-small', loc=2)
    plt.title('f M, f B | M = {:g} | B = {:g} | $\\theta$ = {:.0f}'.format(fmass*M, dB.to(u.pc), theta), fontsize='medium')
    
    vlabel = ['$\mu_{\\alpha_\star}$ [mas yr$^{-1}$]','$\mu_{\delta}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    ylims = [[-0.5, 0.5], [-0.5, 0.5], [-25,25]]
    ylims = [[-1,1], [-1,1], [-50,50]]
    for i in range(3):
        plt.sca(ax[i+1])
        plt.ylabel('$\Delta$ {}'.format(vlabel[i]))
        plt.ylim(*ylims[i])

    plt.xlabel('$\phi_2$ [deg]')
    
    plt.tight_layout()
    
    if rfig:
        return fig
    else:
        plt.savefig('../plots/scale_VB_th{:03d}_{:.1f}_{:.1f}.png'.format(th, fv, fb))


########################
# Orbital intersections

def compile_classicals():
    """Create an input table with 6D positions of the classical dwarfs
    Input data: Gaia DR2 1804.09381, table c4"""
    
    gc_frame = coord.Galactocentric(galcen_distance=8*u.kpc, z_sun=0*u.pc)
    frame_dict0 = gc_frame.__dict__
    old_keys = frame_dict0.keys()
    
    frame_dict = {}
    for k in ['galcen_distance', 'roll', 'galcen_v_sun', 'galcen_coord', 'z_sun']:
        frame_dict[k] = frame_dict0['_{}'.format(k)]
    
    t = Table.read('../data/gdr2_satellites_c4.txt', format='ascii')
    
    x = np.array([t['X']-8, t['Y'], t['Z']])*u.kpc
    v = np.array([t['U'], t['V'], t['W']])*u.km/u.s
    
    for i in range(3):
        v[i] = v[i] + gc_frame.galcen_v_sun.d_xyz[i]
    
    xgal = coord.Galactocentric(x, **frame_dict)
    xeq = xgal.transform_to(coord.ICRS)
    veq_ = gc.vgal_to_hel(xeq, v, galactocentric_frame=gc_frame)
    veq = [None] * 3
    veq[0] = veq_[0].to(u.mas/u.yr)
    veq[1] = veq_[1].to(u.mas/u.yr)
    veq[2] = veq_[2].to(u.km/u.s)
    
    # store observables
    data = {'name': t['Name'], 'ra': xeq.ra, 'dec': xeq.dec, 'distance': xeq.distance, 'pmra': veq[0], 'pmdec': veq[1], 'vr': veq[2]}
    
    tout = Table(data=data, names=('name', 'ra', 'dec', 'distance', 'pmra', 'pmdec', 'vr'))
    tout.pprint()
    tout.write('../data/positions_classical.fits', overwrite=True)
    
def compile_globulars():
    """Create an input table with 6D positions of the globular clusters
    Input data: Gaia DR2 1804.09381, table c3"""
    
    gc_frame = coord.Galactocentric(galcen_distance=8*u.kpc, z_sun=0*u.pc)
    frame_dict0 = gc_frame.__dict__
    old_keys = frame_dict0.keys()
    
    frame_dict = {}
    for k in ['galcen_distance', 'roll', 'galcen_v_sun', 'galcen_coord', 'z_sun']:
        frame_dict[k] = frame_dict0['_{}'.format(k)]
    
    t = Table.read('../data/gdr2_satellites_c3.txt', format='ascii')
    
    x = np.array([t['X']-8, t['Y'], t['Z']])*u.kpc
    v = np.array([t['U'], t['V'], t['W']])*u.km/u.s
    
    for i in range(3):
        v[i] = v[i] + gc_frame.galcen_v_sun.d_xyz[i]
    
    xgal = coord.Galactocentric(x, **frame_dict)
    xeq = xgal.transform_to(coord.ICRS)
    veq_ = gc.vgal_to_hel(xeq, v, galactocentric_frame=gc_frame)
    veq = [None] * 3
    veq[0] = veq_[0].to(u.mas/u.yr)
    veq[1] = veq_[1].to(u.mas/u.yr)
    veq[2] = veq_[2].to(u.km/u.s)
    
    # store observables
    data = {'name': t['Name'], 'ra': xeq.ra, 'dec': xeq.dec, 'distance': xeq.distance, 'pmra': veq[0], 'pmdec': veq[1], 'vr': veq[2]}
    
    tout = Table(data=data, names=('name', 'ra', 'dec', 'distance', 'pmra', 'pmdec', 'vr'))
    tout.pprint()
    tout.write('../data/positions_globular.fits', overwrite=True)

def compile_ufds():
    """Create an input table with 6D positions of the ultra-faint dwarf galaxies
    Input data: Simon 1804.10230, tables 1 & 3"""
    
    t1 = Table.read('../data/simon2018_1.txt', format='ascii')
    t2 = Table.read('../data/simon2018_2.txt', format='ascii')

    data = {'name': t1['Dwarf'], 'ra': t2['ra']*u.deg, 'dec': t2['dec']*u.deg, 'distance': t2['distance']*u.kpc, 'pmra': t1['pmra']*u.mas/u.yr, 'pmdec': t1['pmdec']*u.mas/u.yr, 'vr': t2['vhel']*u.km/u.s}
    
    tout = Table(data=data, names=('name', 'ra', 'dec', 'distance', 'pmra', 'pmdec', 'vr'))
    tout.pprint()
    tout.write('../data/positions_ufd.fits', overwrite=True)

def orbit_cross():
    """Check if satellites crossed GD-1"""
    
    # potential
    ham = gp.Hamiltonian(gp.MilkyWayPotential(nucleus=dict(m=0), halo=dict(c=0.95, m=7E11), bulge=dict(m=4E9), disk=dict(m=5.5e10)))
    gc_frame = coord.Galactocentric(galcen_distance=8*u.kpc, z_sun=0*u.pc)
    
    # orbital solution
    pos = np.load('/home/ana/projects/GD1-DR2/data/gd1_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, 
            pm_phi1_cosphi2=pm1*u.mas/u.yr,
            pm_phi2=pm2*u.mas/u.yr,
            radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    dt = 0.5 * u.Myr
    n_steps = 250
    fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=120)

    # find gap 6D location at present
    gap_phi0 = -40*u.deg
    model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    gap_i = np.abs(model_gd1.phi1.wrap_at(180*u.deg) - gap_phi0).argmin()
    gap_w0 = fit_orbit[gap_i]
    
    # gap orbit
    t1 = 0*u.Myr
    t2 = -1*u.Gyr
    dt = -0.5
    t = np.arange(t1.to(u.Myr).value, t2.to(u.Myr).value+dt, dt)
    gap_orbit = ham.integrate_orbit(gap_w0, dt=dt, t1=t1, t2=t2)
    
    
    # plot relative distances as a function of time
    plt.close()
    plt.figure(figsize=(9,5))
    
    lw = 3

    # show classicals
    tcls = Table.read('../data/positions_classical.fits')
    ra, dec, d, pmra, pmdec, vr = tcls['ra'], tcls['dec'], tcls['distance'], tcls['pmra'], tcls['pmdec'], tcls['vr']
    cs = coord.ICRS(ra=ra, dec=dec, distance=d, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr)
    ws = gd.PhaseSpacePosition(cs.transform_to(gc_frame).cartesian)
    satellite_orbit = ham.integrate_orbit(ws, dt=dt, t1=t1, t2=t2)
    for e in range(len(tcls)):
        if e==0:
            label = 'Classical\ndwarfs'
        else:
            label = ''
        rel_distance = np.linalg.norm(gap_orbit.xyz - satellite_orbit.xyz[:,:,e], axis=0)*gap_orbit.xyz[0].unit
        plt.plot(t, rel_distance, '-', color=mpl.cm.Reds(0.9), alpha=0.5, label=label, lw=lw)
    
    # show ultrafaints
    tufd = Table.read('../data/positions_ufd.fits')
    ra, dec, d, pmra, pmdec, vr = tufd['ra'], tufd['dec'], tufd['distance'], tufd['pmra'], tufd['pmdec'], tufd['vr']
    cs = coord.ICRS(ra=ra, dec=dec, distance=d, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr)
    ws = gd.PhaseSpacePosition(cs.transform_to(gc_frame).cartesian)
    satellite_orbit = ham.integrate_orbit(ws, dt=dt, t1=t1, t2=t2)
    for e in range(len(tufd)):
        if e==0:
            label = 'Ultra-faint\ndwarfs'
        else:
            label = ''
        rel_distance = np.linalg.norm(gap_orbit.xyz - satellite_orbit.xyz[:,:,e], axis=0)*gap_orbit.xyz[0].unit
        plt.plot(t, rel_distance, '-', color=mpl.cm.Reds(0.7), alpha=0.5, label=label, lw=lw)
    
    # show globulars
    tgc = Table.read('../data/positions_globular.fits')
    ra, dec, d, pmra, pmdec, vr = tgc['ra'], tgc['dec'], tgc['distance'], tgc['pmra'], tgc['pmdec'], tgc['vr']
    cs = coord.ICRS(ra=ra, dec=dec, distance=d, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr)
    ws = gd.PhaseSpacePosition(cs.transform_to(gc_frame).cartesian)
    satellite_orbit = ham.integrate_orbit(ws, dt=dt, t1=t1, t2=t2)
    for e in range(len(tgc)):
        if e==0:
            label = 'Globular\nclusters'
        else:
            label = ''
        rel_distance = np.linalg.norm(gap_orbit.xyz - satellite_orbit.xyz[:,:,e], axis=0)*gap_orbit.xyz[0].unit
        plt.plot(t, rel_distance, '-', color=mpl.cm.Reds(0.5), alpha=0.5, label=label, lw=lw)

    plt.plot(t, np.abs(gap_orbit.xyz[2]), '-', color=mpl.cm.Reds(0.3), alpha=0.5, label='Disk', lw=lw, zorder=0)
    #plt.plot(t, np.sqrt(gap_orbit.xyz[0]**2 + gap_orbit.xyz[1]**2), 'r-', alpha=0.2)

    plt.ylim(0.1,200)
    plt.gca().set_yscale('log')
    
    plt.legend(loc=2, fontsize='small', markerscale=2)
    plt.xlabel('Time [Myr]')
    plt.ylabel('Relative distance [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/satellite_distances.png', dpi=200)
    plt.savefig('../paper/satellite_distances.pdf')
