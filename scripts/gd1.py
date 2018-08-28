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
from gala.units import galactic

import scipy.optimize

import interact
import myutils

import pickle

gc_frame_dict = {'galcen_distance':8*u.kpc, 'z_sun':0*u.pc}
gc_frame = coord.Galactocentric(**gc_frame_dict)
ham = gp.Hamiltonian(gp.LogarithmicPotential(v_c=225*u.km/u.s, r_h=0*u.kpc, q1=1, q2=1, q3=1, units=galactic))

def gd1_model():
    """Find a model of GD-1 in a log halo"""
    
    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    t = 56*u.Myr
    n_steps = 1000
    dt = t/n_steps

    fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    model_x = model_gd1.phi1.wrap_at(180*u.deg)
    
    # gap location at the present
    phi1_gap = coord.Angle(-40*u.deg)
    i_gap = np.argmin(np.abs(model_x - phi1_gap))
    out = {'x_gap': fit_orbit.pos.get_xyz()[:,i_gap], 'v_gap': fit_orbit.vel.get_d_xyz()[:,i_gap], 'frame': gc_frame}
    pickle.dump(out, open('../data/gap_present.pkl', 'wb'))
    
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharex=True)

    plt.sca(ax)
    plt.plot(model_x.deg, model_gd1.phi2.deg, 'k-', label='Orbit')
    plt.plot(model_x.deg[i_gap], model_gd1.phi2.deg[i_gap], 'ko', label='Gap')

    plt.legend(fontsize='small')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.ylim(-12,12)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../plots/gd1_orbit.png', dpi=100)

def impact_geometry():
    """"""
    pkl = pickle.load(open('../data/gap_present.pkl', 'rb'))
    
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    t_impact = 0.5*u.Gyr
    dt = 0.5*u.Myr
    n_steps = np.int64(t_impact / dt)

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    
    # gap location at the time of impact
    xgap = x[:,-1]
    vgap = v[:,-1]
    
    i = np.array([1,0,0], dtype=float)
    j = np.array([0,1,0], dtype=float)
    k = np.array([0,0,1], dtype=float)
    
    # find positional plane
    bi = np.cross(j, vgap)
    bi = bi/np.linalg.norm(bi)
    
    bj = np.cross(vgap, bi)
    bj = bj/np.linalg.norm(bj)
    
    # pick b
    bnorm = 0.06*u.kpc
    theta = coord.Angle(90*u.deg)
    b = bnorm*np.cos(theta)*bi + bnorm*np.sin(theta)*bj
    xsub = xgap + b
    
    # find velocity plane
    vi = np.cross(vgap, b)
    vi = vi/np.linalg.norm(vi)
    
    vj = np.cross(b, vi)
    vj = vj/np.linalg.norm(vj)
    
    # pick v
    vnorm = 190*u.km/u.s
    phi = coord.Angle(0*u.deg)
    vsub = vnorm*np.cos(phi)*vi + vnorm*np.sin(phi)*vj
    
    print(np.dot(vgap,b), np.dot(vsub,b))

def stream_timpact():
    """Stream locations at the time of impact"""
    
    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    t_impact = 0.5*u.Gyr
    dt = 0.5*u.Myr
    n_steps = np.int64(t_impact / dt)

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    xend = x[:,-1]
    vend = v[:,-1]
    
    # fine-sampled orbit at the time of impact
    c_impact = coord.Galactocentric(x=xend[0], y=xend[1], z=xend[2], v_x=vend[0], v_y=vend[1], v_z=vend[2], **gc_frame_dict)
    w0_impact = gd.PhaseSpacePosition(c_impact.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    t = 56*u.Myr
    n_steps = 1000
    dt = t/n_steps

    stream = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    stream_impact = ham.integrate_orbit(w0_impact, dt=dt, n_steps=n_steps)
    
    x = fit_orbit.pos.get_xyz()
    sx = stream.pos.get_xyz()
    sx_impact = stream_impact.pos.get_xyz()
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(5.4,10), sharex=True)
    
    label = ['Y', 'Z']
    for i in range(2):
        plt.sca(ax[i])
        plt.plot(x[0], x[i+1], '-', color='0.85', label='Orbit')
        plt.plot(sx[0], sx[i+1], 'k.', label='Present-day stream')
        plt.plot(sx_impact[0], sx_impact[i+1], '.', color='0.4', label='Stream at impact')
        
        plt.xlim(-20,20)
        plt.ylim(-20,20)
        plt.gca().set_aspect('equal')
        plt.ylabel('{} [kpc]'.format(label[i]))
    
    plt.sca(ax[1])
    plt.legend(fontsize='small', loc=1, markerscale=2)
    plt.xlabel('X [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_rewind.png', dpi=100)
