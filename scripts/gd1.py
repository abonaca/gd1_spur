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
#from scipy.optimize import curve_fit

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

def impact_geometry(t_impact=0.5*u.Gyr):
    """"""
    pkl = pickle.load(open('../data/gap_present.pkl', 'rb'))
    
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.05*u.Myr
    n_steps = np.int64(t_impact / dt)

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    
    # gap location at the time of impact
    xgap = x[:,-1]
    vgap = v[:,-1]
    print(xgap)
    
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
    t_impact = 0.2*u.Gyr
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

def save_members():
    """Save a selection of GD-1 members"""
    
    g = Table.read('../data/gd1-with-masks.fits')
    g = g[g['pm_mask'] & g['gi_cmd_mask']]
    
    g.write('../data/members.fits', overwrite=True)

def encounter(bnorm=0.06*u.kpc, bx=0.06*u.kpc, vnorm=200*u.km/u.s, vx=200*u.km/u.s, M=1e7*u.Msun, t_impact=0.5*u.Gyr, point_mass=True, N=1000, verbose=False, fname='gd1_encounter', fig_return=False, fig_annotate=False, model_return=False):
    """Encounter of GD-1 with a massive perturber"""
    
    ########################
    # Perturber at encounter
    
    pkl = pickle.load(open('../data/gap_present.pkl', 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
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
    by = np.sqrt(bnorm**2 - bx**2)
    b = bx*bi + by*bj
    xsub = xgap + b
    
    # find velocity plane
    vi = np.cross(vgap, b)
    vi = vi/np.linalg.norm(vi)
    
    vj = np.cross(b, vi)
    vj = vj/np.linalg.norm(vj)
    
    # pick v
    vy = np.sqrt(vnorm**2 - vx**2)
    vsub = vx*vi + vy*vj
    
    if verbose:
        print(xsub, np.linalg.norm(xsub))
        print(vsub)
    
    #####################
    # Stream at encounter
    
    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
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
    n_steps = N
    dt = t/n_steps

    stream = ham.integrate_orbit(w0_impact, dt=dt, n_steps=n_steps)
    xs = stream.pos.get_xyz()
    vs = stream.vel.get_d_xyz()
    
    #################
    # Encounter setup
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # generate stream model
    if point_mass:
        potential_perturb = 1
        par_perturb = np.array([M.si.value, 0., 0., 0.])
    else:
        potential_perturb = 2
        a = 1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8)
        par_perturb = np.array([M.si.value, a.si.value, 0, 0, 0])
        #result = scipy.integrate.quad(lambda x: bnorm.to(u.pc).value*np.cos(x)**-2 * (a.to(u.pc).value + bnorm.to(u.pc).value*np.cos(x)**-1)**-2, -0.5*np.pi, 0.5*np.pi)
        #m_ = M * 0.5 * a.to(u.pc).value * (2 / a.to(u.pc).value - result[0])
        #print('{:e} {:e}'.format(M, m_))
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].si.value, vs[1].si.value, vs[2].si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    ###########
    # GD-1 data
    
    g = Table.read('../data/members.fits')
    
    rasterized = False
    if fig_return:
        rasterized = True
    
    plt.close()
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, sharey=True)

    plt.sca(ax[0])
    plt.plot(g['phi1'], g['phi2'], 'ko', ms=2.5, alpha=0.7, mec='none', rasterized=rasterized)
    
    plt.ylabel('$\phi_2$ [deg]')
    plt.gca().set_aspect('equal')

    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.', rasterized=rasterized)

    if fig_annotate:
        txt = plt.text(0.02, 0.9, 'M={:g}\nt={}\nb={:.0f} | bi={:.0f}\nv={:.0f} | vi={:.0f}\nvz={:.0f}\nr={:.0f}'.format(M, t_impact, bnorm.to(u.pc), bx.to(u.pc), vnorm, vx, vsub[2], np.linalg.norm(xsub)*xsub.unit), transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
        txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))

    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.xlim(-85,5)
    plt.ylim(-7,5)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    if fig_return:
        return fig, cg
    else:
        plt.savefig('../plots/{}.png'.format(fname), dpi=100)
        if model_return:
            return cg

def halo():
    """A model of a GD-1 encounter with a halo object"""

    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    t_impact = 0.5*u.Gyr
    
    encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fname='gd1_halo', fig_annotate=True, verbose=True)

def disk():
    """A model of GD-1 encounter with a disk object"""
    
    t_impact = 0.1821*u.Gyr
    M = 1e7*u.Msun
    bnorm = 30*u.pc
    vnorm =225*u.km/u.s
    
    #for bx in np.linspace(-bnorm, bnorm, 10):
        #for vx in np.linspace(-vnorm, vnorm, 10):
            #print(bx, vx)
            #encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fname='gd1_disk_{:02.0f}_{:03.0f}'.format(bx.value, vx.value))

    bx = 3.3*u.pc
    vx = -225*u.km/u.s
    encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fname='gd1_disk_{:.1f}_{:.0f}'.format(bx.value, vx.value), verbose=True, fig_annotate=True)

def disk_configurations(nimpact=1):
    """Explore configurations of disk encounters"""
    if nimpact==1:
        t_impact = 0.1821*u.Gyr
    elif nimpact==2:
        t_impact = 0.418*u.Gyr
    elif nimpact==3:
        t_impact = 0.7129*u.Gyr
    
    M = 1e7*u.Msun
    bnorm = 30*u.pc
    vnorm =225*u.km/u.s
    N = 6
    
    pp = PdfPages('../plots/gd1_disk_{}.pdf'.format(nimpact))
    outdict = {'models':[], 'bx': [], 'vx': []}
    for bx in np.linspace(-bnorm, bnorm, N):
        for vx in np.linspace(-vnorm, vnorm, N):
            print(bx, vx)
            fig, cg = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fig_return=True, fig_annotate=True)
            pp.savefig(fig)
            
            outdict['models'] += [cg]
            outdict['bx'] += [bx]
            outdict['vx'] += [vx]
    
    pickle.dump(outdict, open('../data/gd1_disk_{}_coordinates.pkl'.format(nimpact), 'wb'))
    pp.close()

def disk_encounter(nimpact=1, bnorm=30*u.pc, vnorm=225*u.km/u.s, M=1e7*u.Msun, point_mass=True, N=1000, verbose=False):
    """"""
    if nimpact==1:
        t_impact = 0.1821*u.Gyr
    elif nimpact==2:
        t_impact = 0.418*u.Gyr
    elif nimpact==3:
        t_impact = 0.7129*u.Gyr
    
    ########################
    # Perturber at encounter
    
    pkl = pickle.load(open('../data/gap_present.pkl', 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
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
    

    if verbose:
        print(xsub, np.linalg.norm(xsub))
        print(vsub)
    
    #####################
    # Stream at encounter
    
    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
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
    n_steps = N
    dt = t/n_steps

    stream = ham.integrate_orbit(w0_impact, dt=dt, n_steps=n_steps)
    xs = stream.pos.get_xyz()
    vs = stream.vel.get_d_xyz()
    
    #################
    # Encounter setup
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # generate stream model
    if point_mass:
        potential_perturb = 1
        par_perturb = np.array([M.si.value, 0., 0., 0.])
    else:
        potential_perturb = 2
        a = 1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8)
        par_perturb = np.array([M.si.value, a.si.value, 0, 0, 0])
        #result = scipy.integrate.quad(lambda x: bnorm.to(u.pc).value*np.cos(x)**-2 * (a.to(u.pc).value + bnorm.to(u.pc).value*np.cos(x)**-1)**-2, -0.5*np.pi, 0.5*np.pi)
        #m_ = M * 0.5 * a.to(u.pc).value * (2 / a.to(u.pc).value - result[0])
        #print('{:e} {:e}'.format(M, m_))
    
    ###########
    # GD-1 data
    
    g = Table.read('../data/members.fits')
    
    rasterized = False
    #if fig_return:
        #rasterized = True
    
    plt.close()
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, sharey=True)

    plt.sca(ax[0])
    plt.plot(g['phi1'], g['phi2'], 'ko', ms=2.5, alpha=0.7, mec='none', rasterized=rasterized)
    
    plt.ylabel('$\phi_2$ [deg]')
    plt.gca().set_aspect('equal')

    plt.sca(ax[1])
    
    Nb = 5
    for e, bx in enumerate(np.linspace(-bnorm, bnorm, Nb)):
        # pick b
        by = np.sqrt(bnorm**2 - bx**2)
        b = bx*bi + by*bj
        xsub = xgap + b
        
        # find circular velocity at this location
        phi = np.arctan2(xsub[1], xsub[0])
        vsub = vnorm*np.sin(phi)*i - vnorm*np.cos(phi)*j
        
        x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].si.value, vs[1].si.value, vs[2].si.value)
        stream = {}
        stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
        stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
        
        c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
        cg = c.transform_to(gc.GD1)
        
        plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, '.', color='{}'.format(e/Nb), rasterized=rasterized, label='$b_i$ = {:.0f}'.format(bx.to(u.pc)))

    #txt = plt.text(0.02, 0.9, 'M={:g}\nt={}\nb={:.0f} | bi={:.0f}\nr={:.0f}'.format(M, t_impact, bnorm.to(u.pc), bx.to(u.pc), np.linalg.norm(xsub)*xsub.unit), transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    #txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))

    plt.legend(markerscale=2, handlelength=0.5, fontsize='x-small', loc=2)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.xlim(-85,5)
    plt.ylim(-7,5)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('../plots/gd1_gmc_{}.png'.format(nimpact))

# gap width
def gap_profile(t_impact = 0.5*u.Gyr, N=2000):
    """"""
    
    # model
    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    #t_impact = 0.5*u.Gyr
    
    cg = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, N=N, fname='gd1_{:03.0f}'.format(t_impact.to(u.Myr).value), model_return=True)
    phi2_mask = np.abs(cg.phi2)<0.5*u.deg
    
    # data
    g = Table.read('../data/members.fits')
    phi2_mask_data = np.abs(g['phi2'])<0.5
    phi2_mask_back = (g['phi2']<-0.75) & (g['phi2']>-1.75)
    
    bx = np.linspace(-60,-20,20)
    bc = 0.5 * (bx[1:] + bx[:-1])
    density = False
    
    h_data, be = np.histogram(g['phi1'][phi2_mask_data], bins=bx, density=density)
    yerr = np.sqrt(h_data)
    #h_back, be = np.histogram(g['phi1'][phi2_mask_back], bins=bx, density=density)
    #h_data = h_data - h_back
    #print(h_back)
    #med = np.median(h_data)
    #h_data = h_data / med
    #yerr = yerr / med
    
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(180*u.deg).value, bins=bx, density=density)
    #h_model = h_model / np.median(h_model)
    
    ytop = tophat(bc, 40, 20, -40, 7)
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.plot(bc, h_data, 'ko', label='Data')
    plt.errorbar(bc, h_data, yerr=yerr, fmt='none', color='k', label='')
    plt.plot(bc, h_model, 'ko', ms=10, mec='none', alpha=0.5, label='Model')
    plt.plot(bc, ytop, 'r-', label='Top-hat')
    
    plt.legend(fontsize='small')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\\tilde{N}$')
    
    plt.tight_layout()
    plt.savefig('../plots/gap_profile_{:03.0f}.png'.format(t_impact.to(u.Myr).value))

def gap_sizes():
    """"""
    
    times = np.logspace(np.log10(5),np.log10(500),10)*u.Myr
    
    for t in times:
        gap_profile(t_impact=t, N=2000)

def tophat(x, base_level, hat_level, hat_mid, hat_width):
    ret=[]
    for xx in x:
        if hat_mid-hat_width/2. < xx < hat_mid+hat_width/2.:
            ret.append(hat_level)
        else:
            ret.append(base_level)
    return np.array(ret)

def fit_tophat():
    """Find best-fitting top-hat for GD-1 gap at phi1=-40"""

    x = np.arange(-10., 10., 0.01)
    y = tophat(x, 5.0, 1.0, 0.0, 1.0)+np.random.rand(len(x))*0.2-0.1

    guesses = [ [1.0, 5.0, 0.0, 1.0],
                [1.0, 5.0, 0.0, 0.1],
                [1.0, 5.0, 0.0, 2.0] ]

    plt.close()
    plt.figure()
    
    plt.plot(x,y)

    for guess in guesses:
        popt, pcov = scipy.optimize.curve_fit(tophat, x, y, p0=guess)
        print(popt)
        plt.plot( x, tophat(x, popt[0], popt[1], popt[2], popt[3]) )

    plt.show()
