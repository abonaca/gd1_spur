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
from astropy.constants import G, c as c_
from astropy.io import fits

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

import scipy.optimize
import scipy.spatial
import scipy.interpolate
import time
import emcee
import corner

from colossus.cosmology import cosmology
from colossus.halo import concentration

import sys
if sys.version_info < (3, 0, 0):
    import interact
else:
    import interact3 as interact
import myutils

import pickle
import h5py

gc_frame_dict = {'galcen_distance':8*u.kpc, 'z_sun':0*u.pc}
gc_frame = coord.Galactocentric(**gc_frame_dict)
ham = gp.Hamiltonian(gp.LogarithmicPotential(v_c=225*u.km/u.s, r_h=0*u.kpc, q1=1, q2=1, q3=1, units=galactic))
ham_log = gp.Hamiltonian(gp.LogarithmicPotential(v_c=225*u.km/u.s, r_h=0*u.kpc, q1=1, q2=1, q3=1, units=galactic))
ham_mw = gp.Hamiltonian(gp.load('../data/mwpot.yml'))

def gd1_model(pot='log'):
    """Find a model of GD-1 in a log halo"""
    
    if pot=='log':
        ham = ham_log
    elif pot=='mw':
        ham = ham_mw
    
    # load one orbital point
    pos = np.load('../data/{}_orbit.npy'.format(pot))
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
    out = {'x_gap': fit_orbit.pos.get_xyz()[:,i_gap].si, 'v_gap': fit_orbit.vel.get_d_xyz()[:,i_gap].to(u.m/u.s), 'frame': gc_frame}
    print(out['v_gap'].unit.__dict__)
    pickle.dump(out, open('../data/gap_present_{}_python3.pkl'.format(pot), 'wb'))
    print('{} {}\n{}\n{}'.format(i_gap, fit_orbit[i_gap], fit_orbit[0], w0))
    print('dt {}'.format(i_gap*dt.to(u.s)))
    
    out = {'x_gap': fit_orbit.pos.get_xyz()[:,i_gap].si, 'v_gap': fit_orbit.vel.get_d_xyz()[:,i_gap].to(u.m/u.s)}
    tout = Table(out)
    tout.pprint()
    tout.write('../data/gap_present.fits', overwrite=True)
    #h5file = h5py.File('../data/gap_present.h5', 'w')
    #h5file.create_dataset('x_gap', data=fit_orbit.pos.get_xyz()[:,i_gap].si)
    #h5file.create_dataset('v_gap', data=fit_orbit.vel.get_d_xyz()[:,i_gap].si)
    ##h5file.create_dataset("Table2", data=table2, compression=True)
    ## add attributes
    ##h5file["Table2"].attrs["attribute1"] = "some info"
    ##h5file["Table2"].attrs["attribute2"] = 42
    #h5file.close()
    
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
    plt.savefig('../plots/gd1_orbit_{}.png'.format(pot), dpi=100)

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

def encounter(bnorm=0.06*u.kpc, bx=0.06*u.kpc, vnorm=200*u.km/u.s, vx=200*u.km/u.s, M=1e7*u.Msun, rs=0*u.pc, t_impact=0.5*u.Gyr, point_mass=True, N=1000, verbose=False, fname='gd1_encounter', fig_return=False, fig_annotate=False, model_return=False, fig_plot=True):
    """Encounter of GD-1 with a massive perturber"""
    
    ########################
    # Perturber at encounter
    
    pkl = pickle.load(open('../data/gap_present_log_python3.pkl', 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = np.int64((t_impact / dt).decompose())

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
    n_steps = np.int64((t_impact / dt).decompose())

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    xend = x[:,-1]
    vend = v[:,-1]
    
    # fine-sampled orbit at the time of impact
    c_impact = coord.Galactocentric(x=xend[0], y=xend[1], z=xend[2], v_x=vend[0], v_y=vend[1], v_z=vend[2], **gc_frame_dict)
    w0_impact = gd.PhaseSpacePosition(c_impact.transform_to(gc_frame).cartesian)
    
    # stream == best-fitting orbit
    t = 56*u.Myr
    n_steps = N
    dt = t/n_steps

    stream_init = ham.integrate_orbit(w0_impact, dt=dt, n_steps=n_steps)
    xs = stream_init.pos.get_xyz()
    vs = stream_init.vel.get_d_xyz()
    
    #################
    # Encounter setup
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    #T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    #rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.to(u.m/u.s).value, q.value, rhalo.to(u.m).value])
    
    # generate stream model
    if point_mass:
        potential_perturb = 1
        par_perturb = np.array([M.to(u.kg).value, 0., 0., 0.])
    else:
        potential_perturb = 2
        #a = 1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8)
        par_perturb = np.array([M.to(u.kg).value, rs.to(u.m).value, 0, 0, 0])
        #result = scipy.integrate.quad(lambda x: bnorm.to(u.pc).value*np.cos(x)**-2 * (a.to(u.pc).value + bnorm.to(u.pc).value*np.cos(x)**-1)**-2, -0.5*np.pi, 0.5*np.pi)
        #m_ = M * 0.5 * a.to(u.pc).value * (2 / a.to(u.pc).value - result[0])
        #print('{:e} {:e}'.format(M, m_))
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.to(u.m).value, vsub.to(u.m/u.s).value, Tenc.to(u.s).value, t_impact.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, xs[0].to(u.m).value, xs[1].to(u.m).value, xs[2].to(u.m).value, vs[0].to(u.m/u.s).value, vs[1].to(u.m/u.s).value, vs[2].to(u.m/u.s).value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    ###########
    # GD-1 data
    
    g = Table.read('../data/members.fits')
    
    if fig_plot:
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
        return fig, cg, stream_init.energy()
    else:
        plt.savefig('../plots/{}.png'.format(fname), dpi=100)
        if model_return:
            return cg, stream_init.energy()

def halo():
    """A model of a GD-1 encounter with a halo object"""

    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    t_impact = 0.5*u.Gyr
    
    cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fname='gd1_halo', fig_annotate=True, verbose=True, model_return=True)
    outdict = {'model': cg, 'energy': e}
    pickle.dump(outdict, open('../data/gd1_halo_coordinates.pkl', 'wb'))

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
    outdict = {'models':[], 'energy':[], 'bx': [], 'vx': []}
    for bx in np.linspace(-bnorm, bnorm, N):
        for vx in np.linspace(-vnorm, vnorm, N):
            print(bx, vx)
            fig, cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fig_return=True, fig_annotate=True)
            pp.savefig(fig)
            
            outdict['models'] += [cg]
            outdict['energy'] += [e]
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
def stream_track():
    """Show stream track"""
    g = Table.read('../data/members.fits')
    
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
    x = np.linspace(-100,20,100)
    y = poly(x)
    
    plt.close()
    plt.figure(figsize=(12,4))
    
    plt.plot(g['phi1'], g['phi2'], 'k.', ms=1)
    plt.plot(x, y-0.5, 'r-', zorder=0)
    plt.plot(x, y+0.5, 'r-', zorder=0)
    
    plt.xlim(-80,5)
    plt.ylim(-10,5)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()

def gap_profile(t_impact=0.5*u.Gyr, N=2000):
    """"""
    
    # model
    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    #t_impact = 0.5*u.Gyr
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
    
    cg, e_ = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, N=N, fname='gd1_{:03.0f}'.format(t_impact.to(u.Myr).value), model_return=True)
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(180*u.deg).value))<0.5
    
    # data
    g = Table.read('../data/members.fits')
    phi2_mask_data = np.abs(g['phi2'] - poly(g['phi1']))<0.5
    
    bx = np.linspace(-60,-20,30)
    bc = 0.5 * (bx[1:] + bx[:-1])
    Nb = np.size(bc)
    density = False
    
    h_data, be = np.histogram(g['phi1'][phi2_mask_data], bins=bx, density=density)
    yerr_data = np.sqrt(h_data)

    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(180*u.deg).value, bins=bx, density=density)
    yerr_model = np.sqrt(h_data)
    
    # data tophat
    phi1_edges = np.array([-55, -45, -35, -25, -43, -37])
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    data_base = np.median(h_data[base_mask])
    data_hat = np.median(h_data[hat_mask])

    position = -40.5
    width = 8.5
    ytop_data = tophat(bc, data_base, data_hat, position, width)
    np.savez('../data/gap_properties', position=position, width=width, phi1_edges=phi1_edges, yerr=yerr_data)
    
    # model tophat
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    model_hat = np.minimum(model_hat, model_base*0.5)
    ytop_model = tophat(bc, model_base, model_hat, position, width)
    
    chi_gap = np.sum((h_model - ytop_model)**2/yerr_model**2)/Nb
    print(t_impact, chi_gap)
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.plot(bc, h_data, 'ko', label='Data')
    plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='')
    plt.plot(bc, h_model, 'o', ms=10, mec='none', color='0.5', label='Model')
    plt.errorbar(bc, h_model, yerr=yerr_model, fmt='none', color='0.5', label='')
    plt.plot(bc, ytop_data, '-', color='k', alpha=0.5, label='Top-hat data')
    plt.plot(bc, ytop_model, '-', color='0.5', alpha=0.5, label='Top-hat model')
    
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

def visualize_gap_ages():
    """"""
    
    # model
    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
    times = np.logspace(np.log10(5),np.log10(500),15)
    times = np.concatenate([np.array([0]), times])*u.Myr
    N = 2000
    
    #g = Table.read('../data/members.fits')
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))

    for t in times[:]:
        cg, de = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t, N=N, fig_plot=False, model_return=True)
        
        cgal = cg.transform_to(gc_frame)
        outdict = {'stream': cg, 'phi1': cg.phi1.wrap_at(180*u.deg), 'phi2': cg.phi2, 'x': cgal.x, 'y': cgal.y, 'z': cgal.z, 'vx': cgal.v_x, 'vy': cgal.v_y, 'vz': cgal.v_z}
        pickle.dump(outdict, open('../data/snaps/snapshot_{:03.0f}.pkl'.format(t.value), 'wb'))
    
        plt.close()
        fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, sharey=True)

        plt.sca(ax[0])
        plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1)
        
        plt.ylabel('$\phi_2$ [deg]')
        plt.gca().set_aspect('equal')

        plt.sca(ax[1])
        plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.', ms=5)

        plt.text(0.02, 0.8, '{:.0f} Myr'.format(t.to(u.Myr).value), transform=plt.gca().transAxes, fontsize='small', color='0.2')
        #if fig_annotate:
            #txt = plt.text(0.02, 0.9, 'M={:g}\nt={}\nb={:.0f} | bi={:.0f}\nv={:.0f} | vi={:.0f}\nvz={:.0f}\nr={:.0f}'.format(M, t_impact, bnorm.to(u.pc), bx.to(u.pc), vnorm, vx, vsub[2], np.linalg.norm(xsub)*xsub.unit), transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
            #txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))

        plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('$\phi_2$ [deg]')
        
        plt.xlim(-80,0)
        plt.ylim(-7,5)
        plt.gca().set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('../plots/gd1_encounter_{:03.0f}.png'.format(t.value), dpi=170)

def tophat_data(x, hat_mid, hat_width):
    """"""
    base_level = 31.5
    hat_level = 20
    ret=[]
    for xx in x:
        if hat_mid-hat_width/2. < xx < hat_mid+hat_width/2.:
            ret.append(hat_level)
        else:
            ret.append(base_level)
    return np.array(ret)

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


# spuriness

def loop_track():
    """Find track through the observed loop stars"""
    
    g = Table.read('../data/members.fits')
    
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
    x = np.linspace(-50,-39,10)
    y = poly(x)
    
    #x_ = np.array([-36.26, -34.9808, -32.9621, -29.9093])
    #y_ = np.array([1.15983, 1.40602, 1.55373, 1.2583])
    
    x_ = np.array([-38, -36.4793, -34.9808, -32.6, -29.9093])
    y_ = np.array([0.64, 1.01339, 1.04, 1.31276, 1.15])
    
    x_ = np.array([-38, -36.4793, -32.6, -29.9093])
    y_ = np.array([0.64, 1.01339, 1.31276, 1.15])
    
    x = np.concatenate([x, x_])
    y = np.concatenate([y, y_])
    isort = np.argsort(x)
    x = x[isort]
    y = y[isort]
    
    np.savez('../data/spur_track', x=x, y=y)
    
    xi = np.linspace(-50,-30,2000)
    fy = scipy.interpolate.interp1d(x, y, kind='linear')
    #yi = np.interp(xi, x, y)
    yi = fy(xi)
    
    plt.close()
    plt.figure(figsize=(12,5))
    
    plt.plot(g['phi1'], g['phi2'], 'k.', ms=4)
    plt.plot(x, y, 'ro', zorder=0)
    plt.plot(xi, yi, 'r-', lw=0.5)
    
    plt.xlim(-80,0)
    plt.ylim(-10,5)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()

def loop_stars(N=1000, t_impact=0.5*u.Gyr, bnorm=0.06*u.kpc, bx=0.06*u.kpc, vnorm=200*u.km/u.s, vx=200*u.km/u.s, M=1e7*u.Msun):
    """Identify loop stars"""
    
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
    
    Ep = 0.5*(225*u.km/u.s)**2*np.log(np.sum(xs.value**2, axis=0))
    Ek = 0.5*np.sum(vs**2, axis=0)
    Etot = Ep + Ek
    
    Ep_true = stream.potential_energy()
    Etot_true = stream.energy()
    Ek_true = stream.kinetic_energy()
    
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
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])

    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].si.value, vs[1].si.value, vs[2].si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    Ep_stream = 0.5*(225*u.km/u.s)**2*np.log(stream['x'][0].value**2 + stream['x'][1].value**2 + stream['x'][2].value**2)
    Ek_stream = 0.5*(stream['v'][0]**2 + stream['v'][1]**2 + stream['v'][2]**2)
    Etot_stream = Ep_stream + Ek_stream
    
    rE = np.abs(1 - Etot_stream/Etot)
    dE = Etot - Etot_stream
    Ntrue = np.size(rE)
    N2 = int(Ntrue/2)
    
    m1 = np.median(rE[:N2])
    m2 = np.median(rE[N2:])
    
    offset = 0.001
    top1 = np.percentile(dE[:N2], 3)*dE.unit
    top2 = np.percentile(dE[N2:], 92)*dE.unit
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    
    #ind_loop1 = np.where(rE[:N2]>m1+offset)[0][0]
    #ind_loop2 = np.where(rE[N2:]>m2+offset)[0][-1]
    
    print(ind_loop1, ind_loop2)
    
    loop_mask = np.zeros(Ntrue, dtype=bool)
    loop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(180*u.deg)>-50.*u.deg) & (cg.phi1.wrap_at(180*u.deg)<-30*u.deg)
    loop_mask = loop_mask & phi1_mask
    print(np.sum(loop_mask))
    
    # chi spur
    sp = np.load('../data/spur_track.npz')
    f = scipy.interpolate.interp1d(sp['x'], sp['y'], kind='quadratic')
    
    Nloop = np.sum(loop_mask)
    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(180*u.deg).value[loop_mask]))**2/0.5**2)/Nloop
    print(chi_spur)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,7))
    
    plt.sca(ax[0])
    plt.plot(cg.phi1.wrap_at(180*u.deg), dE, 'o')
    plt.plot(cg.phi1.wrap_at(180*u.deg)[loop_mask], dE[loop_mask], 'o')
    
    #plt.plot(cg.phi1.wrap_at(180*u.deg)[N2:], rE[N2:], 'o')
    #plt.plot(cg.phi1.wrap_at(180*u.deg)[:N2], rE[:N2], 'o')
    
    #plt.plot(rE[:N2], 'o')
    #plt.plot(rE[N2:], 'o')
    
    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'o')
    plt.plot(cg.phi1.wrap_at(180*u.deg)[loop_mask], cg.phi2[loop_mask], 'o')
    
    #ls = [':', '-']
    #for e, f in enumerate([0, 0.0002]):
        #plt.axhline(f+m1, ls=ls[e])
        #plt.axhline(f+m2, ls=ls[e], color='tab:orange')
    
    #plt.axvline(ind_loop1)
    #plt.axvline(ind_loop2, color='tab:orange')
    
    plt.xlim(-80,-20)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()


############################
# Mock gap & spur properties

def mock_gap_profile(t_impact=0.5*u.Gyr, N=2000):
    """"""
    
    # model
    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    #t_impact = 0.5*u.Gyr
    #p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    #poly = np.poly1d(p)
    
    # data
    if sys.version_info < (3, 0, 0):
        pkl = pickle.load(open('../data/fiducial.pkl', 'rb'))
    else:
        pkl = pickle.load(open('../data/fiducial_perturb_python3.pkl', 'rb'))
    cg_ = pkl['cg']
    g = {'phi1': cg_.phi1.wrap_at(180*u.deg).value, 'phi2': cg_.phi2.value}
    #g = Table.read('../data/members.fits')
    
    ind = (g['phi1']<0) & (g['phi1']>-80)
    p_mock = np.polyfit(g['phi1'][ind], g['phi2'][ind], deg=4)
    poly = np.poly1d(p_mock)
    
    phi2_mask_data = np.abs(g['phi2'] - poly(g['phi1']))<0.5
    
    bx = np.linspace(-60,-20,30)
    bc = 0.5 * (bx[1:] + bx[:-1])
    Nb = np.size(bc)
    density = False
    
    h_data, be = np.histogram(g['phi1'][phi2_mask_data], bins=bx, density=density)
    yerr_data = np.sqrt(h_data)

    # data tophat
    phi1_edges = np.array([-55, -45, -35, -25, -43, -37])
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    data_base = np.median(h_data[base_mask])
    data_hat = np.median(h_data[hat_mask])

    position = -40.5
    width = 8.5
    ytop_data = tophat(bc, data_base, data_hat, position, width)
    np.savez('../data/mock_gap_properties', position=position, width=width, phi1_edges=phi1_edges, yerr=yerr_data)
    
    #cg, e_ = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, N=N, fname='gd1_{:03.0f}'.format(t_impact.to(u.Myr).value), model_return=True, fig_return=False)
    #phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(180*u.deg).value))<0.5
    
    #h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(180*u.deg).value, bins=bx, density=density)
    #yerr_model = np.sqrt(h_data)
    
    ## model tophat
    #model_base = np.median(h_model[base_mask])
    #model_hat = np.median(h_model[hat_mask])
    #model_hat = np.minimum(model_hat, model_base*0.5)
    #ytop_model = tophat(bc, model_base, model_hat, position, width)
    
    #chi_gap = np.sum((h_model - ytop_model)**2/yerr_model**2)/Nb
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.plot(bc, h_data, 'ko', label='Data')
    plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='')
    #plt.plot(bc, h_model, 'o', ms=10, mec='none', color='0.5', label='Model')
    #plt.errorbar(bc, h_model, yerr=yerr_model, fmt='none', color='0.5', label='')
    plt.plot(bc, ytop_data, '-', color='k', alpha=0.5, label='Top-hat data')
    #plt.plot(bc, ytop_model, '-', color='0.5', alpha=0.5, label='Top-hat model')
    
    plt.legend(fontsize='small')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\\tilde{N}$')
    
    plt.tight_layout()
    plt.savefig('../plots/mock_gap_profile_{:03.0f}.png'.format(t_impact.to(u.Myr).value))

def mock_loop_track():
    """Find track through the observed loop stars"""
    
    if sys.version_info < (3, 0, 0):
        pkl = pickle.load(open('../data/fiducial.pkl', 'rb'))
    else:
        pkl = pickle.load(open('../data/fiducial_perturb_python3.pkl', 'rb'))
    cg_ = pkl['cg']
    g = {'phi1': cg_.phi1.wrap_at(180*u.deg).value, 'phi2': cg_.phi2.value}
    #g = Table.read('../data/members.fits')
    
    #p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    #poly = np.poly1d(p)
    #print(poly)
    
    ind = (g['phi1']<0) & (g['phi1']>-80)
    p_mock = np.polyfit(g['phi1'][ind], g['phi2'][ind], deg=4)
    np.save('../data/mock_polytrack', p_mock)
    poly = np.poly1d(p_mock)
    x = np.linspace(-50,-39,10)
    y = poly(x)
    
    #x_ = np.array([-36.26, -34.9808, -32.9621, -29.9093])
    #y_ = np.array([1.15983, 1.40602, 1.55373, 1.2583])
    
    x_ = np.array([-38, -36.4793, -34.9808, -32.6, -29.9093])
    y_ = np.array([0.64, 1.01339, 1.04, 1.31276, 1.15])
    
    x_ = np.array([-38, -36.4793, -32.6, -29.9093])
    y_ = np.array([0.64, 1.01339, 1.31276, 1.15])
    
    x_ = np.array([-38, -36.5, -34.382, -32.6, -29.9093, -28.3728])
    y_ = np.array([0.482075, 0.869105, 1.11355, 1.31276, 1.7257, 1.8876])
    
    x = np.concatenate([x, x_])
    y = np.concatenate([y, y_])
    isort = np.argsort(x)
    x = x[isort]
    y = y[isort]
    
    np.savez('../data/mock_spur_track', x=x, y=y)
    
    xi = np.linspace(-50,-28.5,2000)
    fy = scipy.interpolate.interp1d(x, y, kind='linear')
    #yi = np.interp(xi, x, y)
    yi = fy(xi)
    
    plt.close()
    plt.figure(figsize=(12,5))
    
    plt.plot(g['phi1'], g['phi2'], 'k.', ms=1)
    plt.plot(x, y, 'ro', zorder=0)
    plt.plot(xi, yi, 'r-', lw=0.5)
    
    plt.xlim(-80,0)
    plt.ylim(-10,5)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()


# search parameter space

def gauge_encounter():
    """"""
    pk0 = pickle.load(open('../data/gd1_halo_coordinates.pkl', 'rb'))
    pk = pickle.load(open('../data/gd1_disk_1_coordinates.pkl', 'rb'))
    pk['models'] += [pk0['model']]
    pk['energy'] += [pk0['energy']]
    Nmod = len(pk['models'])
    
    chi_gap = np.zeros(Nmod)
    chi_spur = np.zeros(Nmod)
    
    sp = np.load('../data/spur_track.npz')
    f = scipy.interpolate.interp1d(sp['x'], sp['y'], kind='quadratic')
    
    bx = np.linspace(-60,-20,30)
    bc = 0.5 * (bx[1:] + bx[:-1])
    Nb = np.size(bc)
    
    gap = np.load('../data/gap_properties.npz')
    phi1_edges = gap['phi1_edges']
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
    
    ind2plot = [9, 12, 17, 36, Nmod]
    cnt = 0
    
    plt.close()
    plt.figure(figsize=(15,6))
    
    ax1 = plt.subplot2grid((4, 3), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((4, 3), (0, 1), colspan=2)
    ax3 = plt.subplot2grid((4, 3), (1, 1), colspan=2, sharex=ax2)
    ax4 = plt.subplot2grid((4, 3), (2, 1), colspan=2, sharex=ax2)
    ax5 = plt.subplot2grid((4, 3), (3, 1), colspan=2, sharex=ax2)
    ax = [ax1, ax2, ax3, ax4, ax5]
    
    for e in range(Nmod):
        cg = pk['models'][e]
        cgal = cg.transform_to(gc_frame)
        
        # spur chi^2
        Ep = 0.5*(225*u.km/u.s)**2*np.log(cgal.x.to(u.kpc).value**2 + cgal.y.to(u.kpc).value**2 + cgal.z.to(u.kpc).value**2)
        Ek = 0.5*(cgal.v_x**2 + cgal.v_y**2 + cgal.v_z**2)
        Etot = Ep + Ek

        dE = pk['energy'][e] - Etot
        N = np.size(dE)
        N2 = int(N/2)

        top1 = np.percentile(dE[:N2], 3)*dE.unit
        top2 = np.percentile(dE[N2:], 92)*dE.unit
        ind_loop1 = np.where(dE[:N2]<top1)[0][0]
        ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
        
        aloop_mask = np.zeros(N, dtype=bool)
        aloop_mask[ind_loop1:ind_loop2+N2] = True
        phi1_mask = (cg.phi1.wrap_at(180*u.deg)>-50.*u.deg) & (cg.phi1.wrap_at(180*u.deg)<-30*u.deg)
        loop_mask = aloop_mask & phi1_mask
        Nloop = np.sum(loop_mask)
        
        chi_spur[e] = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(180*u.deg).value[loop_mask]))**2/0.5**2)/Nloop
        
        # gap chi^2
        phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(180*u.deg).value))<0.5
        h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(180*u.deg).value, bins=bx)
        
        model_base = np.median(h_model[base_mask])
        model_hat = np.median(h_model[hat_mask])
        model_hat = np.minimum(model_hat, model_base*0.5)
        ytop_model = tophat(bc, model_base, model_hat, gap['position'], gap['width'])
        
        chi_gap[e] = np.sum((h_model - ytop_model)**2/gap['yerr']**2)/Nb
        
        if e==ind2plot[cnt]:
            plt.sca(ax[cnt+1])
            plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.')
            
            if cnt<3:
                plt.gca().tick_params(labelbottom=False)
            else:
                plt.xlabel('$\phi_1$ [deg]')
            
            plt.text(0.03, 0.8, '{}'.format(e), fontsize='small', transform=plt.gca().transAxes)
            plt.xlim(-80,0)
            plt.ylabel('$\phi_2$ [deg]')
            
            cnt += 1
    
    plt.sca(ax[0])
    plt.plot(chi_gap, chi_spur, 'ko')
    plt.plot(chi_gap[-1], chi_spur[-1], 'ro')
    
    plt.xlabel('$\chi^2_{gap}$')
    plt.ylabel('$\chi^2_{spur}$')
    
    for e in range(Nmod):
        plt.text(chi_gap[e]*1.07, chi_spur[e]*1.07, '{}'.format(e), fontsize='xx-small')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/encounter_likeness.png')

def generate_slices():
    """"""
    
    t_impact = 0.5*u.Gyr
    M = 7e6*u.Msun
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    
    # fiducial model
    cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, t_impact=t_impact, fname='gd1_halo', fig_annotate=True, verbose=False, model_return=True, fig_plot=False)
    outdict = {'model': [cg], 'energy': [e], 't_impact': [t_impact], 'M': [M], 'vnorm': [vnorm], 'vx': [vx], 'bnorm': [bnorm], 'bx': [bx]}
    
    params0 = [M, vnorm, bnorm]
    params = [M, vnorm, bnorm]
    Npar = len(params)
    flist = np.linspace(0.4,1.6,10)
    
    for ipar in range(Npar):
        for f in flist:
            for i in range(Npar):
                params[i] = params0[i]
            params[ipar] = f * params0[ipar]
            print(params)
            
            cg, e = encounter(bnorm=params[2], bx=params[2], vnorm=params[1], vx=params[1], M=params[0], t_impact=t_impact, fname='gd1_halo', fig_annotate=True, verbose=False, model_return=True, fig_plot=False)
            
            outdict['model'] += [cg]
            outdict['energy'] += [e]
            outdict['t_impact'] += [t_impact]
            outdict['M'] += [params[0]]
            outdict['vnorm'] += [params[1]]
            outdict['vx'] += [params[1]]
            outdict['bnorm'] += [params[2]]
            outdict['bx'] += [params[2]]
    
    pickle.dump(outdict, open('../data/fiducial_slices.pkl', 'wb'))

def slice_likelihood():
    """"""
    pk = pickle.load(open('../data/fiducial_slices.pkl', 'rb'))
    Nmod = len(pk['model'])
    
    chi_gap = np.zeros(Nmod)
    chi_spur = np.zeros(Nmod)
    
    sp = np.load('../data/spur_track.npz')
    f = scipy.interpolate.interp1d(sp['x'], sp['y'], kind='quadratic')
    
    bx = np.linspace(-60,-20,30)
    bc = 0.5 * (bx[1:] + bx[:-1])
    Nb = np.size(bc)
    
    gap = np.load('../data/gap_properties.npz')
    phi1_edges = gap['phi1_edges']
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
    
    ind2plot = [1,10,11,22,Nmod]
    cnt = 0
    
    plt.close()
    plt.figure(figsize=(15,6))
    
    ax1 = plt.subplot2grid((4, 3), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((4, 3), (0, 1), colspan=2)
    ax3 = plt.subplot2grid((4, 3), (1, 1), colspan=2, sharex=ax2)
    ax4 = plt.subplot2grid((4, 3), (2, 1), colspan=2, sharex=ax2)
    ax5 = plt.subplot2grid((4, 3), (3, 1), colspan=2, sharex=ax2)
    ax = [ax1, ax2, ax3, ax4, ax5]
    
    for key in ['M', 'vnorm', 'bnorm']:
        print(pk[key][12])
    
    for e in range(Nmod):
        cg = pk['model'][e]
        cgal = cg.transform_to(gc_frame)
        
        # spur chi^2
        Ep = 0.5*(225*u.km/u.s)**2*np.log(cgal.x.to(u.kpc).value**2 + cgal.y.to(u.kpc).value**2 + cgal.z.to(u.kpc).value**2)
        Ek = 0.5*(cgal.v_x**2 + cgal.v_y**2 + cgal.v_z**2)
        Etot = Ep + Ek

        dE = pk['energy'][e] - Etot
        N = np.size(dE)
        N2 = int(N/2)

        top1 = np.percentile(dE[:N2], 3)*dE.unit
        top2 = np.percentile(dE[N2:], 92)*dE.unit
        ind_loop1 = np.where(dE[:N2]<top1)[0][0]
        ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
        
        aloop_mask = np.zeros(N, dtype=bool)
        aloop_mask[ind_loop1:ind_loop2+N2] = True
        phi1_mask = (cg.phi1.wrap_at(180*u.deg)>-50.*u.deg) & (cg.phi1.wrap_at(180*u.deg)<-30*u.deg)
        loop_mask = aloop_mask & phi1_mask
        Nloop = np.sum(loop_mask)
        
        chi_spur[e] = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(180*u.deg).value[loop_mask]))**2/0.5**2)/Nloop
        
        # gap chi^2
        phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(180*u.deg).value))<0.5
        h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(180*u.deg).value, bins=bx)
        
        model_base = np.median(h_model[base_mask])
        model_hat = np.median(h_model[hat_mask])
        model_hat = np.minimum(model_hat, model_base*0.5)
        ytop_model = tophat(bc, model_base, model_hat, gap_position['position'], gap_width['width'])
        
        chi_gap[e] = np.sum((h_model - ytop_model)**2/gap['yerr']**2)/Nb
        
        if e==ind2plot[cnt]:
            plt.sca(ax[cnt+1])
            plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.')
            plt.plot(cg.phi1.wrap_at(180*u.deg)[loop_mask], cg.phi2[loop_mask], '.', color='tab:orange')
            
            if cnt<3:
                plt.gca().tick_params(labelbottom=False)
            else:
                plt.xlabel('$\phi_1$ [deg]')
            
            plt.text(0.03, 0.8, '{}'.format(e), fontsize='small', transform=plt.gca().transAxes)
            plt.xlim(-80,0)
            plt.ylabel('$\phi_2$ [deg]')
            
            cnt += 1
    
    plt.sca(ax[0])
    plt.plot(chi_gap[1:11], chi_spur[1:11], 'o-', label='M varies')
    plt.plot(chi_gap[11:21], chi_spur[11:21], 'o-', label='v varies')
    plt.plot(chi_gap[21:], chi_spur[21:], 'o-', label='b varies')

    plt.plot(chi_gap[0], chi_spur[0], 'ko', ms=8, label='Fiducial')
    
    plt.legend(fontsize='x-small')
    plt.xlabel('$\chi^2_{gap}$')
    plt.ylabel('$\chi^2_{spur}$')
    
    for e in range(Nmod):
        plt.text(chi_gap[e]+0.02, chi_spur[e]+0.02, '{}'.format(e), fontsize='xx-small')
    
    plt.tight_layout(h_pad=0)


def mw_potential():
    """"""
    ham_mw = gp.Hamiltonian(gp.load('../data/mwpot.yml'))
    print(ham_mw.potential.parameters)
    print(ham_mw.potential)

def test_abinitio(pot='log'):
    """"""
    
    t_impact = 1.5*u.Gyr
    bx = 30*u.pc
    by = 0*u.pc
    vx = 225*u.km/u.s
    vy = 0*u.km/u.s
    M = 1e7*u.Msun
    rs = 10*u.pc
    
    if pot=='log':
        potential = 3
        Vh = 225*u.km/u.s
        q = 1*u.Unit(1)
        rhalo = 0*u.pc
        par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
        ham = ham_log
    elif pot=='mw':
        potential = 6
        Mh = 7e11*u.Msun
        Rh = 15.62*u.kpc
        Vh = np.sqrt(G*Mh/Rh).to(u.km/u.s)
        par_gal = [4e9*u.Msun, 1*u.kpc, 5.5e10*u.Msun, 3*u.kpc, 0.28*u.kpc, Vh, 15.62*u.kpc, 0*u.rad, 1*u.Unit(1), 1*u.Unit(1), 0.95*u.Unit(1)]
        par_pot = np.array([x_.si.value for x_ in par_gal])
        ham = ham_mw
    
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])
    potential_perturb = 2
    par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
    Tenc = 0.12*u.Gyr
    
    pkl = pickle.load(open('../data/gap_present_{}.pkl'.format(pot), 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    xgap = np.array([w0.pos.x.si.value, w0.pos.y.si.value, w0.pos.z.si.value])
    vgap = np.array([w0.vel.d_x.si.value, w0.vel.d_y.si.value, w0.vel.d_z.si.value])
    
    # load orbital end point
    pos = np.load('../data/{}_orbit.npy'.format(pot))
    phi1, phi2, d, pm1, pm2, vr = pos

    c_end = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0_end = gd.PhaseSpacePosition(c_end.transform_to(gc_frame).cartesian)
    xend = np.array([w0_end.pos.x.si.value, w0_end.pos.y.si.value, w0_end.pos.z.si.value])
    vend = np.array([w0_end.vel.d_x.si.value, w0_end.vel.d_y.si.value, w0_end.vel.d_z.si.value])
    
    dt_coarse = 0.5*u.Myr
    Tstream = 56*u.Myr
    Tgap = 29.176*u.Myr
    Nstream = 2000
    dt_stream = Tstream/Nstream
    dt_fine = 0.05*u.Myr

    t1 = time.time()
    x1, x2, x3, v1, v2, v3, de = interact.abinit_interaction(xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Tgap.si.value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    t2 = time.time()
    
    ########################
    # Perturber at encounter
    
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
    
    j = np.array([0,1,0], dtype=float)
    
    # find positional plane
    bi = np.cross(j, vgap)
    bi = bi/np.linalg.norm(bi)
    
    bj = np.cross(vgap, bi)
    bj = bj/np.linalg.norm(bj)
    
    b = bx*bi + by*bj
    xsub = xgap + b
    
    # find velocity plane
    vi = np.cross(vgap, b)
    vi = vi/np.linalg.norm(vi)
    
    vj = np.cross(b, vi)
    vj = vj/np.linalg.norm(vj)
    
    # pick v
    vsub = vx*vi + vy*vj
    
    print(xsub.si)
    print(vsub.si)
    
    #####################
    # Stream at encounter
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = np.int64(t_impact / dt)

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0_end, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    xend = x[:,-1]
    vend = v[:,-1]
    
    #print(xend.si, vend.si)
    #print(fit_orbit.energy()[0]/fit_orbit.energy()[-1])
    
    # fine-sampled orbit at the time of impact
    c_impact = coord.Galactocentric(x=xend[0], y=xend[1], z=xend[2], v_x=vend[0], v_y=vend[1], v_z=vend[2], **gc_frame_dict)
    w0_impact = gd.PhaseSpacePosition(c_impact.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    stream_init = ham.integrate_orbit(w0_impact, dt=dt_stream, n_steps=Nstream)
    xs = stream_init.pos.get_xyz()
    vs = stream_init.vel.get_d_xyz()
    
    x1_, x2_, x3_, v1_, v2_, v3_ = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt_fine.si.value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].si.value, vs[1].si.value, vs[2].si.value)
    
    t3 = time.time()
    
    dt_c = t2 - t1
    dt_p = t3 - t2
    #print(dt_c, dt_p, dt_p/dt_c)
    
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    stream_ = {}
    stream_['x'] = (np.array([x1_, x2_, x3_])*u.m).to(u.kpc)
    stream_['v'] = (np.array([v1_, v2_, v3_])*u.m/u.s).to(u.km/u.s)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    plt.sca(ax[0])
    plt.plot(stream['x'][0], stream['x'][1], 'o', ms=8)
    plt.plot(stream_['x'][0], stream_['x'][1], 'o', ms=6)
    
    plt.sca(ax[1])
    plt.plot(stream['v'][0], stream['v'][1], 'o', ms=8)
    plt.plot(stream_['v'][0], stream_['v'][1], 'o', ms=6)
    
    plt.tight_layout()

import copy
def run(cont=False, steps=100, nwalkers=100, nth=8, label='', potential_perturb=1, test=False, mock=True):
    """"""
    if sys.version_info < (3, 0, 0):
        pkl = pickle.load(open('../data/gap_present.pkl', 'rb'))
    else:
        pkl = pickle.load(open('../data/gap_present_log_python3.pkl', 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    xgap = np.array([w0.pos.x.si.value, w0.pos.y.si.value, w0.pos.z.si.value])
    vgap = np.array([w0.vel.d_x.to(u.m/u.s).value, w0.vel.d_y.to(u.m/u.s).value, w0.vel.d_z.to(u.m/u.s).value])
    
    # load orbital end point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c_end = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0_end = gd.PhaseSpacePosition(c_end.transform_to(gc_frame).cartesian)
    xend = np.array([w0_end.pos.x.si.value, w0_end.pos.y.si.value, w0_end.pos.z.si.value])
    vend = np.array([w0_end.vel.d_x.to(u.m/u.s).value, w0_end.vel.d_y.to(u.m/u.s).value, w0_end.vel.d_z.to(u.m/u.s).value])
    
    dt_coarse = 0.5*u.Myr
    Tstream = 56*u.Myr
    Tgap = 29.176*u.Myr
    Nstream = 2000
    N2 = int(Nstream*0.5)
    dt_stream = Tstream/Nstream
    dt_fine = 0.05*u.Myr
    wangle = 180*u.deg
    
    if mock:
        prefix = 'mock_'
        label = '_mock' + label
    else:
        prefix = ''
    
    
    # gap comparison
    bins = np.linspace(-60,-20,30)
    bc = 0.5 * (bins[1:] + bins[:-1])
    Nb = np.size(bc)
    Nside_min = 5
    f_gap = 0.5
    delta_phi2 = 0.5
    
    gap = np.load('../data/{:s}gap_properties.npz'.format(prefix))
    phi1_edges = gap['phi1_edges']
    gap_position = gap['position']
    gap_width = gap['width']
    gap_yerr = gap['yerr']
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    
    p = np.load('../data/{:s}polytrack.npy'.format(prefix))
    poly = np.poly1d(p)
    x_ = np.linspace(-100,0,100)
    y_ = poly(x_)
    
    # spur comparison
    sp = np.load('../data/{:s}spur_track.npz'.format(prefix))
    spx = sp['x']
    spy = sp['y']
    phi2_err = 0.2
    phi1_min = -50*u.deg
    phi1_max = -30*u.deg
    percentile1 = 3
    percentile2 = 92
    quad_phi1 = -32*u.deg
    quad_phi2 = 0.8*u.deg
    Nquad = 1
    
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.to(u.m/u.s).value, q.value, rhalo.si.value])
    
    Tenc = 0.01*u.Gyr
    #potential_perturb = 1
    #par_perturb = np.array([M.si.value, 0., 0., 0.])
    #potential_perturb = 2
    #par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
    
    chigap_max = 0.6567184385873621
    chispur_max = 1.0213837095314207
    
    chigap_max = 0.8
    chispur_max = 1.2
    
    # parameters to sample
    t_impact = 0.5*u.Gyr
    bx = 40*u.pc
    by = 1*u.pc
    vx = 225*u.km/u.s
    vy = 1*u.km/u.s
    M = 7e6*u.Msun
    rs = 0.5*u.pc
    #print((2*G*M*c_**-2).to(u.pc))
    #print(1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8))
    
    if potential_perturb==1:
        params_list = [t_impact, bx, by, vx, vy, M, Tgap]
    elif potential_perturb==2:
        params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
    params_units = [p_.unit for p_ in params_list]
    params = [p_.value for p_ in params_list]
    params[5] = np.log10(params[5])
    
    model_args = [params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb]
    gap_args = [poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, Nside_min, f_gap, gap_position, gap_width]
    spur_args = [N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad]
    lnp_args = [chigap_max, chispur_max]
    lnprob_args = model_args + gap_args + spur_args + lnp_args
    
    ndim = len(params)
    if cont==False:
        seed = 614398
        np.random.seed(seed)
        p0 = [np.random.randn(ndim) for i in range(nwalkers)]
        p0 = (np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim))*1e-4 + 1.)*np.array(params)[np.newaxis,:]
        
        seed = 3465
        prng = np.random.RandomState(seed)
        genstate = np.random.get_state()
    else:
        rgstate = pickle.load(open('../data/state{}.pkl'.format(label), 'rb'))
        genstate = rgstate['state']
        
        smp = np.load('../data/samples{}.npz'.format(label))
        flatchain = smp['chain']
        chain = np.transpose(flatchain.reshape(nwalkers, -1, ndim), (1,0,2))
        nstep = np.shape(chain)[0]
        flatchain = chain.reshape(nwalkers*nstep, ndim)
        
        positions = np.arange(-nwalkers, 0, dtype=np.int64)
        p0 = flatchain[positions]
    
    if test:
        N = np.size(p0[:,0])
        lnp = np.zeros(N)
        
        for i in range(N):
            args = copy.deepcopy(lnprob_args[:])
            lnp[i] = lnprob(p0[i], *args)
        
        print(N, np.sum(np.isfinite(lnp)))
        
        return
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nth, args=lnprob_args, runtime_sortingfn=sort_on_runtime)
    
    t1 = time.time()
    pos, prob, state = sampler.run_mcmc(p0, steps, rstate0=genstate)
    t2 = time.time()
    
    if cont==False:
        np.savez('../data/samples{}'.format(label), lnp=sampler.flatlnprobability, chain=sampler.flatchain, nwalkers=nwalkers)
    else:
        np.savez('../data/samples{}_temp'.format(label), lnp=sampler.flatlnprobability, chain=sampler.flatchain, nwalkers=nwalkers)
        np.savez('../data/samples{}'.format(label), lnp=np.concatenate([smp['lnp'], sampler.flatlnprobability]), chain=np.concatenate([smp['chain'], sampler.flatchain]), nwalkers=nwalkers)
    
    rgstate = {'state': state}
    pickle.dump(rgstate, open('../data/state{}.pkl'.format(label), 'wb'))
    
    print('Chain: {:5.2f} s'.format(t2 - t1))
    print('Average acceptance fraction: {}'.format(np.average(sampler.acceptance_fraction[0])))
    
    sampler.pool.terminate()

def sort_on_runtime(p):
    """Improve runtime by starting longest jobs first (sorts on first parameter -- in our case, the encounter time)"""
    
    p = np.atleast_2d(p)
    idx = np.argsort(p[:, 0])[::-1]
    
    return p[idx], idx

def lnprob(x, params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb, poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, Nside_min, f_gap, gap_position, gap_width, N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad, chigap_max, chispur_max):
    """Check if a model is better than the fiducial"""
    
    if (x[0]<0) | (x[0]>14) | (np.sqrt(x[3]**2 + x[4]**2)>500):
        return -np.inf
    
    x[5] = 10**x[5]
    params = [x_*u_ for x_, u_ in zip(x, params_units)]
    if potential_perturb==1:
        t_impact, bx, by, vx, vy, M, Tgap = params
        par_perturb = np.array([M.to(u.kg).value, 0., 0., 0.])
    else:
        t_impact, bx, by, vx, vy, M, rs, Tgap = params
        par_perturb = np.array([M.to(u.kg).value, rs.to(u.m).value, 0., 0., 0.])
        if x[6]<0:
            return -np.inf
    
    if (Tgap<0*u.Myr) | (Tgap>Tstream):
        return -np.inf
    
    # calculate model
    x1, x2, x3, v1, v2, v3, dE = interact.abinit_interaction(xend, vend, dt_coarse.to(u.s).value, dt_fine.to(u.s).value, t_impact.to(u.s).value, Tenc.to(u.s).value, Tstream.to(u.s).value, Tgap.to(u.s).value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.to(u.m).value, by.to(u.m).value, vx.to(u.m/u.s).value, vy.to(u.m/u.s).value)
    
    c = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    # spur chi^2
    top1 = np.percentile(dE[:N2], percentile1)
    top2 = np.percentile(dE[N2:], percentile2)
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    
    f = scipy.interpolate.interp1d(spx, spy, kind='quadratic')
    
    aloop_mask = np.zeros(Nstream, dtype=bool)
    aloop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(wangle)>phi1_min) & (cg.phi1.wrap_at(wangle)<phi1_max)
    loop_mask = aloop_mask & phi1_mask
    Nloop = np.sum(loop_mask)
    
    loop_quadrant = (cg.phi1.wrap_at(wangle)[loop_mask]>quad_phi1) & (cg.phi2[loop_mask]>quad_phi2)
    if np.sum(loop_quadrant)<Nquad:
        return -np.inf
    
    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(wangle).value[loop_mask]))**2/phi2_err**2)/Nloop
    
    # gap chi^2
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(wangle).value))<delta_phi2
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(wangle).value, bins=bins)
    yerr = np.sqrt(h_model+1)
    
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    #model_hat = np.minimum(model_hat, model_base*f_gap)
    if (model_base<Nside_min) | (model_hat>model_base*f_gap):
        return -np.inf
    
    ytop_model = tophat(bc, model_base, model_hat,  gap_position, gap_width)
    chi_gap = np.sum((h_model - ytop_model)**2/yerr**2)/Nb
    
    if np.isfinite(chi_gap) & np.isfinite(chi_spur):
        return -(chi_gap + chi_spur)
    else:
        return -np.inf
    #if (chi_gap<chigap_max) & (chi_spur<chispur_max):
        #return 0.
    #else:
        #return -np.inf


def plot_corner(label='', full=False):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = sampler['chain']
    Npar = np.shape(chain)[1]
    print(np.sum(np.isfinite(sampler['lnp'])), np.size(sampler['lnp']))
    
    params = ['T', 'bx', 'by', 'vx', 'vy', 'logM', 'rs', 'Tgap']
    if full==False:
        params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', 'log M/M$_\odot$']
        abr = chain[:,:-3]
        abr[:,1] = np.sqrt(chain[:,1]**2 + chain[:,2]**2)
        abr[:,2] = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
        abr[:,0] = chain[:,0]
        abr[:,3] = chain[:,5]
        if Npar>7:
            abr[:,3] = chain[:,6]
            abr[:,4] = chain[:,5]
            params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', '$r_s$ [pc]', 'log M/M$_\odot$']
            #lims = [[0.,2], [0.1,100], [10,1000], [0.001,1000], [5,9]]
        chain = abr
    
    plt.close()
    corner.corner(chain, bins=50, labels=params, plot_datapoints=True)
    
    plt.savefig('../plots/corner{}{:d}.png'.format(label, full))

def plot_chains(label=''):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = sampler['chain']
    lnp = sampler['lnp']
    nwalkers = sampler['nwalkers']
    ntot, Npar = np.shape(chain)
    nstep = int(ntot/nwalkers)
    steps = np.arange(nstep)
    
    Npanel = Npar + 1
    nrow = np.int(np.ceil(np.sqrt(Npanel)))
    ncol = np.int(np.ceil(Npanel/nrow))
    da = 2.5
    params = ['T [Gyr]', '$B_x$ [pc]', '$B_y$ [pc]', '$V_x$ [km s$^{-1}$]', '$V_y$ [km s$^{-1}$]', 'log M/M$_\odot$', '$r_s$ [pc]', '$T_{gap}$ [Myr]']
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(1.5*ncol*da, nrow*da), sharex=True)
    
    for i in range(Npar):
        plt.sca(ax[int(i/nrow)][i%nrow])
        plt.plot(steps, chain[:,i].reshape(nstep,-1), '-', rasterized=True)
        plt.ylabel(params[i])
    
    plt.sca(ax[nrow-1][ncol-1])
    plt.plot(steps, lnp.reshape(nstep,-1), '-', rasterized=True)
    plt.ylabel('ln P')
    
    for i in range(ncol):
        plt.sca(ax[nrow-1][i])
        plt.xlabel('Step')
        
    plt.tight_layout()
    plt.savefig('../plots/chain{}.png'.format(label))

def trim_chain(chain, nwalkers, nstart, npar, nend=-1):
    """Trim number of usable steps in a chain"""
    
    chain = chain.reshape(nwalkers,-1,npar)
    chain = chain[:,nstart:nend,:]
    chain = chain.reshape(-1, npar)
    
    return chain

def trim_lnp(lnp, nwalkers, nstart):
    """Trim number of usable steps in lnp"""
    
    lnp = lnp.reshape(nwalkers,-1)
    lnp = lnp[:,nstart:]
    lnp = lnp.reshape(-1)
    
    return lnp

def explore_islands(label='', n=1):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = sampler['chain']
    
    params = ['T', 'bx', 'by', 'vx', 'vy', 'logM', 'rs']
    B = np.sqrt(chain[:,1]**2 + chain[:,2]**2)
    V = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
    
    if n==1:
        island = (chain[:,0]>0.7) & (chain[:,0]<1.) & (V<500)
    elif n==2:
        island = (chain[:,0]>1) & (chain[:,0]<1.5)
    else:
        island = (chain[:,0]>1.5)
    Nisland = np.sum(island)
    Nc = 10
    np.random.seed(59)
    ind = np.random.randint(Nisland, size=Nc)

    for k in ind:
        x = chain[island][k]
        lnprob_args = get_lnprobargs()
        params_units = lnprob_args[0]
        fig, ax, chi_gap, chi_spur, N = lnprob_verbose(x, *lnprob_args)
        plt.suptitle('  '.join(['{:.2g} {}'.format(x_, u_) for x_, u_ in zip(x, params_units)]), fontsize='medium')
        plt.tight_layout(rect=[0,0,1,0.96])
        
        plt.savefig('../plots/likelihood_island{}_{}.png'.format(n, k))

def get_unique(label=''):
    """Save unique models in a separate file"""
    
    sampler = np.load('../data/samples{}.npz'.format(label))
    models, ind = np.unique(sampler['chain'], axis=0, return_index=True)
    #print(np.shape(models), np.shape(ind))
    #print(np.shape(np.unique(sampler['lnp'])))
    ifinite = np.isfinite(sampler['lnp'][ind])
    
    np.savez('../data/unique_samples{}'.format(label), chain=models[ifinite], lnp=sampler['lnp'][ind][ifinite])

def choose_lnp_threshold(label='', p=10):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    ifinite = np.isfinite(sampler['lnp'])
    lnp = sampler['lnp'][ifinite]
    
    pp = np.percentile(lnp, p)
    print(pp)
    
    plt.close()
    plt.figure(figsize=(8,5))
    
    plt.hist(lnp, bins=100, histtype='step', color='k', lw=2)
    plt.axvline(pp, color='firebrick', lw=4, alpha=0.4)
    
    plt.gca().set_yscale('log')
    
    plt.tight_layout()

def check_chain(full=False, label='', p=5):
    """"""
    sampler = np.load('../data/unique_samples{}.npz'.format(label))
    
    models = np.unique(sampler['chain'], axis=0)
    models = sampler['chain']
    lnp = sampler['lnp']
    pp = np.percentile(lnp, p)
    
    ind = lnp>=pp
    models = models[ind]
    lnp = lnp[ind]
    
    params = ['T', 'bx', 'by', 'vx', 'vy', 'logM', 'rs']
    print(np.shape(models), np.shape(models)[0]/np.shape(sampler['chain'])[0])
    Npar = np.shape(models)[1]
    
    if full==False:
        abr = models[:,:-3]
        abr[:,1] = np.sqrt(models[:,1]**2 + models[:,2]**2)
        abr[:,2] = np.sqrt(models[:,3]**2 + models[:,4]**2)
        abr[:,0] = models[:,0]
        abr[:,3] = models[:,5]
        print(np.median(abr[:,1]), np.max(abr[:,1]))
        params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', 'log M/M$_\odot$']
        
        if Npar>6:
            abr[:,3] = models[:,6]
            abr[:,4] = models[:,5]
            params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', '$r_s$ [pc]', 'log M/M$_\odot$']
            lims = [[0.,2], [0.1,75], [10,500], [0.001,40], [5,9]]
            lims = [[0.,4.5], [0,145], [0,700], [0,99], [4.5,9]]
            logscale = [False, True, True, True, False]
            logscale = [False, False, False, False, False]
        else:
            lims = [[0.,2], [0.1,100], [10,1000], [5,9]]
            logscale = [False, True, True, False]
        models = abr
    
    Nvar = np.shape(models)[1]
    dax = 2
    
    Nsample = np.shape(models)[0]
    Nc = 20
    np.random.seed(59)
    ind = np.random.randint(Nsample, size=Nc)
    hull_ids = np.empty(0, dtype=int)
    panel_id = np.empty((0,2), dtype=int)
    vertices = np.empty((0,2))

    plt.close()
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row' ,squeeze=False)
    
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar):
            plt.sca(ax[j-1][i])
            
            #plt.plot(models[:,i], models[:,j], '.', ms=1, alpha=0.1, color='0.2', rasterized=True)
            
            #for k in range(Nc):
                #plt.plot(models[ind[k]][i], models[ind[k]][j], 'o', ms=4, color='PaleVioletRed')
            
            points = np.log10(np.array([models[:,i], models[:,j]]).T)
            hull = scipy.spatial.ConvexHull(points)
            
            xy_vert = 10**np.array([points[hull.vertices,0], points[hull.vertices,1]]).T
            vertices = np.concatenate([vertices, xy_vert])
            hull_ids = np.concatenate([hull_ids, hull.vertices])
            
            #print(len(hull.vertices))
            current_id = np.tile(np.array([i,j]), len(hull.vertices)).reshape(-1,2)
            panel_id = np.concatenate([panel_id, current_id])
            
            p = mpl.patches.Polygon(xy_vert, closed=True, lw=2, ec='0.8', fc='0.9', zorder=0)
            plt.gca().add_artist(p)
            
            #for simplex in hull.simplices:
                #plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
    #hull_ids = np.unique(hull_ids)
    #print(np.size(hull_ids))
    np.savez('../data/hull_points{}'.format(label), all=hull_ids, unique=np.unique(hull_ids), panel=panel_id, vertices=vertices)
    
    t_impact = 0.495*u.Gyr
    M = 5e6*u.Msun
    rs = 0.1*rs_diemer(M)
    bnorm = 15*u.pc
    vnorm = 250*u.km/u.s
    
    pfid = [t_impact.to(u.Gyr).value, bnorm.to(u.pc).value, vnorm.to(u.km/u.s).value, (rs.to(u.pc).value), np.log10(M.to(u.Msun).value)]
    
    for i in range(Nvar-1):
        for j in range(i+1, Nvar):
            #ind = i + (j-1)*Nvar + Nvar
            plt.sca(ax[j-1][i])
            plt.plot(pfid[i], pfid[j], '*', ms=20, mec='orangered', mew=1.5, color='orange')
    
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar-1):
            plt.sca(ax[i][j])
            plt.axis('off')
    
    for k in range(Nvar-1):
        plt.sca(ax[-1][k])
        plt.xlabel(params[k])

        if full==False:
            if logscale[k]:
                plt.gca().set_xscale('log')
            plt.xlim(lims[k])
        
        plt.sca(ax[k][0])
        plt.ylabel(params[k+1])
        if (full==False):
            if logscale[k+1]:
                plt.gca().set_yscale('log')
            plt.ylim(lims[k+1])
    
    #if Npar>6:
        #mrange = 10**np.linspace(np.min(models[:,4]), np.max(models[:,4]), 20)*u.Msun
        #rsrange = rs_hernquist(mrange)
        #rsrange2 = rs_diemer(mrange)
        
        #plt.sca(ax[3][3])
        #plt.plot(0.1*rsrange2.to(u.pc), np.log10(mrange.value), ':', color='DarkSlateBlue', lw=1.5)
        #plt.plot(0.3*rsrange2.to(u.pc), np.log10(mrange.value), '--', color='DarkSlateBlue', lw=1.5)
        #plt.plot(0.84*rsrange2.to(u.pc), np.log10(mrange.value), '-', color='DarkSlateBlue', lw=1.5)
        #plt.plot(1.16*rsrange2.to(u.pc), np.log10(mrange.value), '-', color='DarkSlateBlue', lw=1.5)

    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/corner{}_f{:d}.png'.format(label, full))
    plt.savefig('../paper/corner.pdf')

def check_impulse(label=''):
    """"""
    
    x = np.load('../data/unique_samples{}.npz'.format(label))['chain']
    M = 10**x[:,5] * u.Msun
    B = np.sqrt(x[:,1]**2 + x[:,2]**2)*u.pc
    V = np.sqrt(x[:,3]**2 + x[:,4]**2)*u.km/u.s
    
    f = (G*M/(B*V**2)).decompose()
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    
    plt.sca(ax[0][0])
    plt.hist(M, bins=np.logspace(5.5,8.5,20))
    plt.gca().set_xscale('log')
    
    plt.xlabel('log M/M$_\odot$')
    plt.ylabel('N')
    
    plt.sca(ax[0][1])
    plt.hist(B, bins=np.logspace(0,2,20))
    plt.gca().set_xscale('log')
    
    plt.xlabel('B [pc]')
    plt.ylabel('N')
    
    plt.sca(ax[1][0])
    plt.hist(V, bins=np.logspace(1.5,3,20))
    plt.gca().set_xscale('log')
    
    plt.xlabel('V [km s$^{-1}$]')
    plt.ylabel('N')
    
    plt.sca(ax[1][1])
    plt.hist(f, bins=np.logspace(-3,1,20))
    plt.gca().set_xscale('log')
    
    plt.xlabel('$f_{impulse}$ = GM/BV$^2$')
    plt.ylabel('N')
    
    plt.tight_layout()
    plt.savefig('../plots/impulse{}.png'.format(label))

def lnprob_verbose(x, params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb, poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, f_gap, gap_position, gap_width, N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad, chigap_max, chispur_max, colored=True, plot_comp=True, chi_label=True):
    """Check if a model is better than the fiducial"""
    
    if (x[0]<0) | (np.sqrt(x[3]**2 + x[4]**2)>1000):
        return -np.inf
    
    x[5] = 10**x[5]
    params = [x_*u_ for x_, u_ in zip(x, params_units)]

    if potential_perturb==1:
        t_impact, bx, by, vx, vy, M, Tgap = params
        par_perturb = np.array([M.si.value, 0., 0., 0.])
        par_noperturb = np.array([0., 0., 0., 0.])
    else:
        t_impact, bx, by, vx, vy, M, rs, Tgap = params
        par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
        par_noperturb = np.array([0., rs.si.value, 0., 0., 0.])
    
    # calculate model
    #x1, x2, x3, v1, v2, v3, dE = interact.abinit_interaction(xgap, vgap, xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    x1, x2, x3, v1, v2, v3, dE = interact.abinit_interaction(xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Tgap.si.value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    c = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    #x1, x2, x3, v1, v2, v3, dE_ = interact.abinit_interaction(xgap, vgap, xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Nstream, par_pot, potential, par_noperturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    #c_np = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    #cg_np = c_np.transform_to(gc.GD1)
    
    
    # gap chi^2
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(wangle).value))<delta_phi2
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(wangle).value, bins=bins)
    yerr = np.sqrt(h_model+1)
    
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    model_hat = np.minimum(model_hat, model_base*f_gap)
    ytop_model = tophat(bc, model_base, model_hat,  gap_position, gap_width)
    
    chi_gap = np.sum((h_model - ytop_model)**2/(yerr)**2)/Nb
    
    # spur chi^2
    top1 = np.percentile(dE[:N2], percentile1)
    top2 = np.percentile(dE[N2:], percentile2)
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    
    f = scipy.interpolate.interp1d(spx, spy, kind='quadratic')
    
    aloop_mask = np.zeros(Nstream, dtype=bool)
    aloop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(wangle)>phi1_min) & (cg.phi1.wrap_at(wangle)<phi1_max)
    loop_mask = aloop_mask & phi1_mask
    Nloop = np.sum(loop_mask)

    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(wangle).value[loop_mask]))**2/phi2_err**2)/Nloop

    loop_quadrant = (cg.phi1.wrap_at(wangle)[loop_mask]>quad_phi1) & (cg.phi2[loop_mask]>quad_phi2)
    print(chi_gap, chi_spur, np.sum(loop_quadrant))
    
    plt.close()
    fig, ax = plt.subplots(3,2,figsize=(13,9))
    
    plt.sca(ax[0][0])
    plt.plot(bc, h_model, 'o')
    if plot_comp:
        plt.plot(bc, ytop_model, 'k-')

    if chi_label:
        plt.text(0.95, 0.15, '$\chi^2_{{gap}}$ = {:.2f}'.format(chi_gap), ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.ylabel('N')
    plt.xlim(-60,-20)
    
    plt.sca(ax[1][0])
    plt.plot(cg.phi1.wrap_at(wangle).value, dE, 'o')
    if colored:
        plt.plot(cg.phi1.wrap_at(wangle).value[aloop_mask], dE[aloop_mask], 'o')
    plt.ylabel('$\Delta$ E')
    
    plt.sca(ax[2][0])
    plt.plot(cg.phi1.wrap_at(wangle).value, cg.phi2.value, 'o')
    if colored:
        plt.plot(cg.phi1.wrap_at(wangle).value[loop_mask], cg.phi2.value[loop_mask], 'o')
    if plot_comp:
        isort = np.argsort(cg.phi1.wrap_at(wangle).value[loop_mask])
        plt.plot(cg.phi1.wrap_at(wangle).value[loop_mask][isort], f(cg.phi1.wrap_at(wangle).value[loop_mask])[isort], 'k-')
    
    if chi_label:
        plt.text(0.95, 0.15, '$\chi^2_{{spur}}$ = {:.2f}'.format(chi_spur), ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    plt.xlim(-60,-20)
    plt.ylim(-5,5)
    
    plt.sca(ax[0][1])
    plt.plot(c.x.to(u.kpc), c.y.to(u.kpc), 'o')
    if colored:
        plt.plot(c.x.to(u.kpc)[loop_mask], c.y.to(u.kpc)[loop_mask], 'o') #, color='orange')
    
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    
    plt.sca(ax[1][1])
    cr = np.sqrt(c.x**2 + c.y**2)
    plt.plot(cr.to(u.kpc), c.z.to(u.kpc), 'o')
    if colored:
        plt.plot(cr.to(u.kpc)[loop_mask], c.z.to(u.kpc)[loop_mask], 'o') #, color='orange')
    
    plt.xlabel('R [kpc]')
    plt.ylabel('z [kpc]')
    
    plt.sca(ax[2][1])
    isort = np.argsort(cg.phi1.wrap_at(wangle).value[~aloop_mask])
    vr0 = np.interp(cg.phi1.wrap_at(wangle).value, cg.phi1.wrap_at(wangle).value[~aloop_mask][isort], cg.radial_velocity.to(u.km/u.s)[~aloop_mask][isort])*u.km/u.s
    dvr = vr0 - cg.radial_velocity.to(u.km/u.s)
    #plt.plot(cg.phi1.wrap_at(wangle).value, cg.radial_velocity.to(u.km/u.s), 'o')
    #plt.plot(cg.phi1.wrap_at(wangle).value[~aloop_mask], cg.radial_velocity.to(u.km/u.s)[~aloop_mask], 'o')
    #plt.plot(cg.phi1.wrap_at(wangle).value, vr0, 'o')
    plt.plot(cg.phi1.wrap_at(wangle).value, dvr, 'o')
    if colored:
        plt.plot(cg.phi1.wrap_at(wangle).value[loop_mask], dvr[loop_mask], 'o')
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
    plt.ylim(-5,5)
    plt.xlim(-60,-20)
    
    plt.tight_layout()
    
    return fig, ax, chi_gap, chi_spur, np.sum(loop_quadrant), -(chi_gap + chi_spur)

def lnprob_detailed(x, params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb, poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, f_gap, gap_position, gap_width, N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad, chigap_max, chispur_max, colored=True, plot_comp=True, chi_label=True):
    """Check if a model is better than the fiducial"""
    
    if (x[0]<0) | (np.sqrt(x[3]**2 + x[4]**2)>1000):
        return -np.inf
    
    x[5] = 10**x[5]
    params = [x_*u_ for x_, u_ in zip(x, params_units)]

    if potential_perturb==1:
        t_impact, bx, by, vx, vy, M, Tgap = params
        par_perturb = np.array([M.si.value, 0., 0., 0.])
        par_noperturb = np.array([0., 0., 0., 0.])
    else:
        t_impact, bx, by, vx, vy, M, rs, Tgap = params
        par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
        par_noperturb = np.array([0., rs.si.value, 0., 0., 0.])
    
    # calculate model
    x1, x2, x3, v1, v2, v3, dE = interact.abinit_interaction(xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Tgap.si.value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    c = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    # gap chi^2
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(wangle).value))<delta_phi2
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(wangle).value, bins=bins)
    yerr = np.sqrt(h_model+1)
    
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    model_hat = np.minimum(model_hat, model_base*f_gap)
    ytop_model = tophat(bc, model_base, model_hat,  gap_position, gap_width)
    
    chi_gap = np.sum((h_model - ytop_model)**2/(yerr)**2)/Nb
    
    # spur chi^2
    top1 = np.percentile(dE[:N2], percentile1)
    top2 = np.percentile(dE[N2:], percentile2)
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    print(ind_loop1, cg.phi1.wrap_at(wangle).deg[ind_loop1])
    print(ind_loop2, cg.phi1.wrap_at(wangle).deg[ind_loop2+N2])
    
    f = scipy.interpolate.interp1d(spx, spy, kind='quadratic')
    
    aloop_mask = np.zeros(Nstream, dtype=bool)
    aloop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(wangle)>phi1_min) & (cg.phi1.wrap_at(wangle)<phi1_max)
    loop_mask = aloop_mask & phi1_mask
    Nloop = np.sum(loop_mask)
    #print(np.min(cg.phi1.wrap_at(wangle).deg[loop]))

    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(wangle).value[loop_mask]))**2/phi2_err**2)/Nloop

    loop_quadrant = (cg.phi1.wrap_at(wangle)[loop_mask]>quad_phi1) & (cg.phi2[loop_mask]>quad_phi2)
    
    isort = np.argsort(cg.phi1.wrap_at(wangle).value[~aloop_mask])
    vr0 = np.interp(cg.phi1.wrap_at(wangle).value, cg.phi1.wrap_at(wangle).value[~aloop_mask][isort], cg.radial_velocity.to(u.km/u.s)[~aloop_mask][isort])*u.km/u.s
    dvr = cg.radial_velocity.to(u.km/u.s) - vr0
    
    mu10 = np.interp(cg.phi1.wrap_at(wangle).value, cg.phi1.wrap_at(wangle).value[~aloop_mask][isort], cg.pm_phi1_cosphi2.to(u.mas/u.yr)[~aloop_mask][isort])*u.mas/u.yr
    dmu1 = cg.pm_phi1_cosphi2.to(u.mas/u.yr) - mu10
    
    mu20 = np.interp(cg.phi1.wrap_at(wangle).value, cg.phi1.wrap_at(wangle).value[~aloop_mask][isort], cg.pm_phi2.to(u.mas/u.yr)[~aloop_mask][isort])*u.mas/u.yr
    dmu2 = cg.pm_phi2.to(u.mas/u.yr) - mu20
    
    dist0 = np.interp(cg.phi1.wrap_at(wangle).value, cg.phi1.wrap_at(wangle).value[~aloop_mask][isort], cg.distance.to(u.pc)[~aloop_mask][isort])*u.pc
    ddist = cg.distance.to(u.pc) - dist0
    
    res = {'params': params, 'stream': cg, 'dvr': dvr, 'dmu1': dmu1, 'dmu2': dmu2, 'ddist': ddist, 'all_loop': aloop_mask, 'phi1_loop': loop_mask, 'chi_gap': chi_gap, 'chi_spur': chi_spur, 'bincen': bc, 'nbin': h_model, 'nexp': ytop_model}
    
    return res

def rs_hernquist(M):
    """Return Hernquist scale radius for a halo of mass M"""
    
    return 1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8)

def rs_diemer(M):
    """Return NFW scale radius for a halo of mass M, assuming Diemer+2018 mass-concetration relation"""
    
    cosmology.setCosmology('planck15')
    csm = cosmology.getCurrent()
    
    z_ = 0
    rho_c = csm.rho_c(z_)
    h_ = csm.Hz(z_) * 1e-2
    delta = 200
    
    c, mask = concentration.concentration(M.to(u.Msun).value/h_, '200c', 0.0, model='diemer19', range_return=True)
    R = ((3*M.to(u.Msun).value/h_)/(4*np.pi*delta*rho_c))**(1/3) * h_
    rs = R / c * 1e3 * u.pc
    
    return rs

def rs_moline(M, r=20*u.kpc, Mhost=1e12*u.Msun, verbose=False):
    """Return NFW scale radius for subhalo of mass M at radius r from the center of the host halo of mass Mhost, assuming Moline+2017 mass-concentration relation"""
    
    cosmology.setCosmology('planck15')
    csm = cosmology.getCurrent()
    
    z_ = 0
    h_ = csm.Hz(z_) * 1e-2
    rho_c = csm.rho_c(z_)*h_**2
    delta = 200
    
    c0 = 19.9
    a = np.array([-0.195, 0.089, 0.089])
    b = -0.54
    i = np.array([1, 2, 3])
    
    Rhost = ((3*Mhost.to(u.Msun).value)/(4*np.pi*delta*rho_c))**(1/3) * u.kpc
    rrel = (r / Rhost).decompose()
    
    if verbose: print(rrel, r, Rhost)

    c = c0 * (1 + np.sum( (a[:,np.newaxis] * np.log10((M[np.newaxis,:]).to(u.Msun).value*1e-8))**i[:,np.newaxis], axis=0 )) * (1 + b*np.log10(rrel))
    R = ((3*M.to(u.Msun).value)/(4*np.pi*delta*rho_c))**(1/3)
    rs = R / c * 1e3 * u.pc
    
    if verbose: print(rho_c, R, c)
    
    return rs

def get_lnprobargs():
    """"""
    pkl = pickle.load(open('../data/gap_present_log_python3.pkl', 'rb'))
    c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    xgap = np.array([w0.pos.x.si.value, w0.pos.y.si.value, w0.pos.z.si.value])
    vgap = np.array([w0.vel.d_x.si.value, w0.vel.d_y.si.value, w0.vel.d_z.si.value])
    
    # load orbital end point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c_end = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0_end = gd.PhaseSpacePosition(c_end.transform_to(gc_frame).cartesian)
    xend = np.array([w0_end.pos.x.si.value, w0_end.pos.y.si.value, w0_end.pos.z.si.value])
    vend = np.array([w0_end.vel.d_x.si.value, w0_end.vel.d_y.si.value, w0_end.vel.d_z.si.value])
    
    dt_coarse = 0.5*u.Myr
    Tstream = 56*u.Myr
    Tgap = 29.176*u.Myr
    Nstream = 2000
    N2 = int(Nstream*0.5)
    dt_stream = Tstream/Nstream
    dt_fine = 0.05*u.Myr
    wangle = 180*u.deg
    
    
    # gap comparison
    bins = np.linspace(-60,-20,30)
    bc = 0.5 * (bins[1:] + bins[:-1])
    Nb = np.size(bc)
    f_gap = 0.5
    delta_phi2 = 0.5
    
    gap = np.load('../data/gap_properties.npz')
    phi1_edges = gap['phi1_edges']
    gap_position = gap['position']
    gap_width = gap['width']
    gap_yerr = gap['yerr']
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    
    p = np.load('../data/polytrack.npy')
    poly = np.poly1d(p)
    x_ = np.linspace(-100,0,100)
    y_ = poly(x_)
    
    # spur comparison
    sp = np.load('../data/spur_track.npz')
    spx = sp['x']
    spy = sp['y']
    phi2_err = 0.5
    phi1_min = -50*u.deg
    phi1_max = -30*u.deg
    percentile1 = 3
    percentile2 = 92
    quad_phi1 = -32*u.deg
    quad_phi2 = 0.8*u.deg
    Nquad = 1
    
    # parameters to sample
    t_impact = 0.5*u.Gyr
    bx = 40*u.pc
    by = 1*u.pc
    vx = 225*u.km/u.s
    vy = 1*u.km/u.s
    M = 7e6*u.Msun
    rs = 0*u.pc
    #print((2*G*M*c_**-2).to(u.pc))
    #print(1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8))
    
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    potential_perturb = 2
    #par_perturb = np.array([M.si.value, 0., 0., 0.])
    #potential_perturb = 2
    #par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
    Tenc = 0.01*u.Gyr
    
    chigap_max = 0.6567184385873621
    chispur_max = 1.0213837095314207
    
    params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
    params_units = [p_.unit for p_ in params_list]
    params = [p_.value for p_ in params_list]
    params[5] = np.log10(params[5])
    
    model_args = [params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb]
    gap_args = [poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, f_gap, gap_position, gap_width]
    spur_args = [N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad]
    lnp_args = [chigap_max, chispur_max]
    lnprob_args = model_args + gap_args + spur_args + lnp_args
    
    return lnprob_args

def check_model(fiducial=False, label='', rand=False, Nc=10, fast=True):
    """"""
    chain = np.load('../data/unique_samples{}.npz'.format(label))['chain']
    vnorm = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
    if fast:
        ind = vnorm>490
    else:
        ind = vnorm<350
    chain = chain[ind]
    Nsample = np.shape(chain)[0]
    if rand:
        np.random.seed(59)
        ind = np.random.randint(Nsample, size=Nc)
    else:
        ind = np.load('../data/hull_points{}.npy'.format(label))
    Nc = np.size(ind)
    
    for k in range(Nc):
        x = chain[ind[k]]

        if fiducial:
            t_impact = 0.5*u.Gyr
            bx = 40*u.pc
            by = 0*u.pc
            vx = 225*u.km/u.s
            vy = 0*u.km/u.s
            M = 7e6*u.Msun
            rs = 1*u.pc
            
            t_impact = 0.5*u.Gyr
            bx = 20*u.pc
            by = 0*u.pc
            vx = 50*u.km/u.s
            vy = 10*u.km/u.s
            M = 1e5*u.Msun
            rs = 20*u.pc
            
            print(rs_hernquist(M))
            
            params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
            params_units = [p_.unit for p_ in params_list]
            x = [p_.value for p_ in params_list]
            x[5] = np.log10(x[5])
        
        pkl = pickle.load(open('../data/gap_present.pkl', 'rb'))
        c = coord.Galactocentric(x=pkl['x_gap'][0], y=pkl['x_gap'][1], z=pkl['x_gap'][2], v_x=pkl['v_gap'][0], v_y=pkl['v_gap'][1], v_z=pkl['v_gap'][2], **gc_frame_dict)
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        xgap = np.array([w0.pos.x.si.value, w0.pos.y.si.value, w0.pos.z.si.value])
        vgap = np.array([w0.vel.d_x.si.value, w0.vel.d_y.si.value, w0.vel.d_z.si.value])
        
        # load orbital end point
        pos = np.load('../data/log_orbit.npy')
        phi1, phi2, d, pm1, pm2, vr = pos

        c_end = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
        w0_end = gd.PhaseSpacePosition(c_end.transform_to(gc_frame).cartesian)
        xend = np.array([w0_end.pos.x.si.value, w0_end.pos.y.si.value, w0_end.pos.z.si.value])
        vend = np.array([w0_end.vel.d_x.si.value, w0_end.vel.d_y.si.value, w0_end.vel.d_z.si.value])
        
        dt_coarse = 0.5*u.Myr
        Tstream = 56*u.Myr
        Nstream = 2000
        N2 = int(Nstream*0.5)
        dt_stream = Tstream/Nstream
        dt_fine = 0.05*u.Myr
        wangle = 180*u.deg
        
        
        # gap comparison
        bins = np.linspace(-60,-20,30)
        bc = 0.5 * (bins[1:] + bins[:-1])
        Nb = np.size(bc)
        f_gap = 0.5
        delta_phi2 = 0.5
        
        gap = np.load('../data/gap_properties.npz')
        phi1_edges = gap['phi1_edges']
        gap_position = gap['position']
        gap_width = gap['width']
        gap_yerr = gap['yerr']
        base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
        hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
        
        p = np.load('../data/polytrack.npy')
        poly = np.poly1d(p)
        x_ = np.linspace(-100,0,100)
        y_ = poly(x_)
        
        # spur comparison
        sp = np.load('../data/spur_track.npz')
        spx = sp['x']
        spy = sp['y']
        phi2_err = 0.5
        phi1_min = -50*u.deg
        phi1_max = -30*u.deg
        percentile1 = 3
        percentile2 = 92
        quad_phi1 = -32*u.deg
        quad_phi2 = 0.8*u.deg
        Nquad = 1
        
        # parameters to sample
        t_impact = 0.5*u.Gyr
        bx = 40*u.pc
        by = 1*u.pc
        vx = 225*u.km/u.s
        vy = 1*u.km/u.s
        M = 7e6*u.Msun
        rs = 0*u.pc
        Tgap = 29*u.Myr
        #print((2*G*M*c_**-2).to(u.pc))
        #print(1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8))
        
        potential = 3
        Vh = 225*u.km/u.s
        q = 1*u.Unit(1)
        rhalo = 0*u.pc
        par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
        
        potential_perturb = 2
        #par_perturb = np.array([M.si.value, 0., 0., 0.])
        #potential_perturb = 2
        #par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
        Tenc = 0.01*u.Gyr
        
        chigap_max = 0.6567184385873621
        chispur_max = 1.0213837095314207
        
        params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
        params_units = [p_.unit for p_ in params_list]
        params = [p_.value for p_ in params_list]
        params[5] = np.log10(params[5])
        
        model_args = [params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb]
        gap_args = [poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, f_gap, gap_position, gap_width]
        spur_args = [N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad]
        lnp_args = [chigap_max, chispur_max]
        lnprob_args = model_args + gap_args + spur_args + lnp_args
        
        #print(lnprob(x, *lnprob_args))
        fig, ax, chi_gap, chi_spur, N, lnp = lnprob_verbose(x, *lnprob_args)
        print(lnp)
        
        plt.suptitle('  '.join(['{:.2g} {}'.format(x_, u_) for x_, u_ in zip(x,params_units)]), fontsize='medium')
        plt.tight_layout(rect=[0,0,1,0.96])
        
        plt.savefig('../plots/likelihood_f{:d}_r{:d}_{}.png'.format(fast, rand, k))
    

########################
# Explore fiducial model

def model_time(T, bnorm=0.03*u.kpc, bx=0.03*u.kpc, vnorm=225*u.km/u.s, vx=225*u.km/u.s, M=7e6*u.Msun, t_impact=0.5*u.Gyr):
    """Return model encounter at a time T"""
    
    ########################
    # Perturber at encounter
    
    pkl = pickle.load(open('../data/gap_present_log_python3.pkl', 'rb'))
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
    n_steps = 1000
    dt = t/n_steps

    stream_init = ham.integrate_orbit(w0_impact, dt=dt, n_steps=n_steps)
    xs = stream_init.pos.get_xyz()
    vs = stream_init.vel.get_d_xyz()
    
    #######
    # setup
    
    #################
    # Encounter setup
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    #T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # perturber parameters
    potential_perturb = 2
    rs = 0*u.pc
    #a = 1.05*u.kpc * np.sqrt(M.to(u.Msun).value*1e-8)
    par_perturb = np.array([M.si.value, rs.si.value, 0, 0, 0])
    
    # generate stream model
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, T.si.value, dt.si.value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].si.value, vs[1].si.value, vs[2].si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    
    return c

def model_evolution():
    """"""
    # fiducial setup
    bnorm = 0.03*u.kpc
    bx = 0.03*u.kpc
    vnorm = 225*u.km/u.s
    vx = 225*u.km/u.s
    M = 7e6*u.Msun
    t_impact = 0.5*u.Gyr
    
    da = 2
    Nsnap = 7
    t = np.logspace(-1,0.5,Nsnap)*u.Gyr
    #t = np.linspace(0.01,10,Nsnap)*u.Gyr
    
    pp = PdfPages('../plots/evolution_v.pdf')

    #for bx in np.linspace(-0.03,0.03,7)*u.kpc:
    #for vx in np.linspace(-225,225,7)*u.km/u.s:
    #for m in np.linspace(2e6,12e6,7)[:]*u.Msun:
    #for b in np.linspace(0.01,0.1,7)*u.kpc:
    for v in np.linspace(50,350,7)*u.km/u.s:
        print(v)
        plt.close()
        fig, ax = plt.subplots(2, Nsnap, figsize=(da*Nsnap, da*2), sharex='col')
        
        for i in range(Nsnap):
            c = model_time(t[i], vx=v, vnorm=v)
            
            plt.sca(ax[0][i])
            plt.plot(c.x, c.y, 'k.', ms=1, rasterized=True)
            
            plt.title('T = {:.2f} Gyr'.format(t[i].to(u.Gyr).value), fontsize='medium')
            if i==0:
                plt.ylabel('y [kpc]')
            
            
            plt.sca(ax[1][i])
            plt.plot(c.x, c.z, 'k.', ms=1, rasterized=True)
            
            plt.xlabel('x [kpc]')
            if i==0:
                plt.ylabel('z [kpc]')
        
        #plt.suptitle('M = {:.2g} M$_\odot$'.format(m.value), fontsize='medium')
        plt.suptitle('$V$ = {:.0f} km s$^{{-1}}$'.format(v.value), fontsize='medium')
        #plt.suptitle('$b$ = {:.0f} pc'.format(b.to(u.pc).value), fontsize='medium')
        plt.tight_layout(rect=(0,0,1,0.92))
        pp.savefig(fig)
        #plt.savefig('../plots/evolution_m{:02.1f}.png'.format(m.to(u.Msun).value*1e-6))
    
    pp.close()

def new_fiducial(label='_hernquist_pl'):
    """"""
    
    #sampler = np.load('../data/samples{}.npz'.format(label))
    sampler = np.load('../data/unique_samples{}.npz'.format(label))
    chain = sampler['chain']
    lnp = sampler['lnp']
    
    Npar = np.shape(chain)[1]
    params = ['T', 'bx', 'by', 'vx', 'vy', 'logM', 'rs']
    
    v = np.sqrt(chain[:,3]**2 + chain[:,4]**2) * u.km/u.s
    m = 10**chain[:,5] * u.Msun
    rs = chain[:,6] * u.pc
    
    rmin = 0.1 * rs_diemer(m)
    vmin = 400 * u.km/u.s
    
    keep = (rs>rmin) & (v<vmin) & (chain[:,4]<0.5) & (chain[:,2]<0.5)
    samples = chain[keep]
    lnp_samples = lnp[keep]
    
    keep_top = lnp_samples>np.percentile(lnp_samples, 95)
    samples = samples[keep_top]
    lnp_samples = lnp_samples[keep_top]
    
    #indmax = np.argmin(lnp_samples)
    #fid = samples[indmax]
    fid = np.median(samples, axis=0)
    print(fid)
    
    t_impact = fid[0] * u.Gyr
    bx = fid[1] * u.pc
    bnorm = np.sqrt(fid[1]**2 + fid[2]**2) * u.pc
    vx = fid[3] * u.km/u.s
    vnorm = np.sqrt(fid[3]**2 + fid[4]**2) * u.km/u.s
    M = 10**fid[5] * u.Msun
    rs = fid[6] * u.pc
    
    cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t_impact, fname='gd1_fiducial', point_mass=False, fig_annotate=True, verbose=True, model_return=True)
    

def manual_fiducial():
    """A model of a GD-1 encounter with a halo object"""

    bnorm = 10*u.pc
    bx = 10*u.pc
    vnorm = 300*u.km/u.s
    vx = -300*u.km/u.s
    M = 6e6*u.Msun
    t_impact = 0.5*u.Gyr
    rs = 12*u.pc
    
    print(rs_diemer(M))
    cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t_impact, N=3000, fname='gd1_manfid', point_mass=False, fig_annotate=True, verbose=True, model_return=True)


def orbit_fiducial(phi=0, theta=180):
    """Find orbit alternative to the streakline fiducial"""
    
    bnorm = 15*u.pc
    bx = bnorm * np.cos(np.radians(phi))
    vnorm = 250*u.km/u.s
    vx = vnorm * np.cos(np.radians(theta))
    M = 5e6*u.Msun
    t_impact = 0.495*u.Gyr
    rs = 0.1*rs_diemer(M)
    
    cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t_impact, N=1000, fname='gd1_manfid', point_mass=False, fig_annotate=True, verbose=True, model_return=True)

def model_examples(model=0, i=0, label='_v500w200', verbose=False):
    """Pick several models that showcase the allowed diversity of impactor parameters"""
    
    if model==0:
        phi = -20
        theta = 170
        bnorm = 15*u.pc
        bx = bnorm * np.cos(np.radians(phi))
        by = bnorm * np.sin(np.radians(phi))
        vnorm = 250*u.km/u.s
        vx = vnorm * np.cos(np.radians(theta))
        vy = vnorm * np.sin(np.radians(theta))
        M = 5e6*u.Msun
        t_impact = 0.495*u.Gyr
        rs = 0.1*rs_diemer(M)
        Tgap = 29.176*u.Myr
        i = -1
        
        params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
        params_units = [p_.unit for p_ in params_list]
        x = [p_.value for p_ in params_list]
        x[5] = np.log10(x[5])
    else:
        np.random.seed(4619)
        chain = np.load('../data/unique_samples{}.npz'.format(label))['chain']
        Nsample = np.shape(chain)[0]
        ind = np.random.randint(Nsample, size=i+1)
        x = chain[ind][-1]
    
    bnorm = np.sqrt(x[1]**2 + x[2]**2)
    vnorm = np.sqrt(x[3]**2 + x[4]**2)
    if verbose:
        print(i, x)
    
    lnprob_args = get_lnprobargs()
    spx = lnprob_args[-7]
    spy = lnprob_args[-6]
    #lnprob_args[7] = 3000
    #lnprob_args[-12] = 10
    #lnprob_args[-11] = 99
    wangle = 180*u.deg
    
    res = lnprob_detailed(x, *lnprob_args)
    res['x'] = x
    pickle.dump(res, open('../data/predictions/model3_{:03d}.pkl'.format(i), 'wb'))
    cg = res['stream']
    
    plt.close()
    fig, ax = plt.subplots(5,1,figsize=(8,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'ko')
    plt.plot(spx, spy, 'r-', alpha=0.5, lw=3)
    
    plt.xlim(-60, -20)
    plt.ylim(-4,4)
    plt.ylabel('$\phi_2$ [deg]')
    plt.text(0.95, 0.15, '$\chi_{{spur}}$ = {:.2f}'.format(res['chi_spur']), transform=plt.gca().transAxes, ha='right', fontsize='small')
    plt.title('log M/M$_\odot$ = {:.2f} | $r_s$ = {:.1f}pc | b = {:.1f}pc | V = {:.0f}km s$^{{-1}}$ | T = {:.2}Gyr'.format(np.log10(x[5]), x[6], bnorm, vnorm, x[0]), fontsize='small')
    
    plt.sca(ax[1])
    plt.plot(res['bincen'], res['nbin'], 'ko')
    plt.plot(res['bincen'], res['nexp'], 'r-', alpha=0.5, lw=3)
    plt.ylabel('Number')
    plt.text(0.95, 0.15, '$\chi_{{gap}}$ = {:.2f}'.format(res['chi_gap']), transform=plt.gca().transAxes, ha='right', fontsize='small')
    
    plt.sca(ax[2])
    plt.plot(cg.phi1.wrap_at(wangle), res['dvr'], 'ko')
    plt.ylim(-5, 5)
    plt.ylabel('$\Delta$ $V_r$\n[km s$^{-1}$]')
    
    plt.sca(ax[3])
    plt.plot(cg.phi1.wrap_at(wangle), res['dmu1'], 'ko')
    plt.ylim(-0.5, 0.5)
    plt.ylabel('$\Delta$ $\mu_{\phi_1}$\n[mas yr$^{-1}$]')
    
    plt.sca(ax[4])
    plt.plot(cg.phi1.wrap_at(wangle), res['dmu2'], 'ko')
    plt.ylim(-0.5, 0.5)
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $\mu_{\phi_2}$\n[mas yr$^{-1}$]')
    
    plt.tight_layout(h_pad=0.06)
    plt.savefig('../plots/predictions/model_{:03d}.png'.format(i))
    

###################
# Streakline stream
from gala.dynamics import mockstream

def generate_streakline_fiducial():
    """"""
    
    np.random.seed(143531)
    
    t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    t_impact = 0.495*u.Gyr
    M = 5e6*u.Msun
    #rs = 0.1*rs_diemer(M)
    rs = 10*u.pc
    bnorm = 15*u.pc
    bx = 6*u.pc
    vnorm = 250*u.km/u.s
    vx = -25*u.km/u.s

    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = 120
    wangle = 180*u.deg

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    
    prog_phi0 = -20*u.deg

    model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    prog_i = np.abs(model_gd1.phi1.wrap_at(180*u.deg) - prog_phi0).argmin()
    prog_w0 = fit_orbit[prog_i]
    
    dt_orbit = 0.5*u.Myr
    nstep_impact = np.int64((t_impact / dt_orbit).decompose())
    #prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
    prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
    impact_orbit = prog_orbit[nstep_impact:]
    impact_orbit = impact_orbit[::-1]
    prog_orbit = prog_orbit[::-1]
    
    t_disrupt = -300*u.Myr
    minit = 7e4
    mfin = 1e3
    nrelease = 1
    n_times = (prog_orbit.t < t_disrupt).sum()
    prog_mass = np.linspace(minit, mfin, n_times)
    prog_mass = np.concatenate((prog_mass, np.zeros(len(prog_orbit.t) - n_times))) * u.Msun
    model_present = mockstream.dissolved_fardal_stream(ham, prog_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)
    
    n_steps_disrupt = int(abs(t_disrupt / (prog_orbit.t[1]-prog_orbit.t[0])))
    model_present = model_present[:-2*n_steps_disrupt]
    
    model_gd1 = model_present.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    ind_gap = np.where((model_gd1.phi1.wrap_at(wangle)>-43*u.deg) & (model_gd1.phi1.wrap_at(wangle)<-33*u.deg))[0]

    n_times = (impact_orbit.t < t_disrupt).sum()
    prog_mass = np.linspace(minit, mfin, n_times)
    prog_mass = np.concatenate((prog_mass, np.zeros(len(impact_orbit.t) - n_times))) * u.Msun
    model = mockstream.dissolved_fardal_stream(ham, impact_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)

    n_steps_disrupt = int(abs(t_disrupt / (impact_orbit.t[1]-impact_orbit.t[0])))
    model = model[:-2*n_steps_disrupt]

    Nstar = np.shape(model)[0]
    ivalid = ind_gap < Nstar
    ind_gap = ind_gap[ivalid]
    
    xgap = np.median(model.xyz[:,ind_gap], axis=1)
    vgap = np.median(model.v_xyz[:,ind_gap], axis=1)
    
    
    ########################
    # Perturber at encounter
    
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
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    dt = 0.05*u.Myr
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # generate unperturbed stream model
    potential_perturb = 2
    par_perturb = np.array([0*M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    wangle = 180*u.deg
    outdict = {'cg': cg}
    #pickle.dump(outdict, open('../data/fiducial_noperturb_python3.pkl', 'wb'))
    
    # generate perturbed stream model
    potential_perturb = 2
    par_perturb = np.array([M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    wangle = 180*u.deg
    outdict = {'cg': cg}
    #pickle.dump(outdict, open('../data/fiducial_perturb_python3.pkl', 'wb'))
    
    plt.close()
    plt.figure(figsize=(10,5))
    plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.')
    plt.xlim(-80,0)
    plt.ylim(-10,10)
    plt.tight_layout()

def streakline_input():
    """Create streakline model of a correct age"""
    
    np.random.seed(143531)
    
    t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    t_impact = 0.495*u.Gyr
    M = 5e6*u.Msun
    #rs = 0.1*rs_diemer(M)
    rs = 10*u.pc
    bnorm = 15*u.pc
    bx = 6*u.pc
    vnorm = 250*u.km/u.s
    vx = -25*u.km/u.s

    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = 120
    wangle = 180*u.deg

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    
    prog_phi0 = -20*u.deg

    model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    prog_i = np.abs(model_gd1.phi1.wrap_at(180*u.deg) - prog_phi0).argmin()
    prog_w0 = fit_orbit[prog_i]
    
    dt_orbit = 0.5*u.Myr
    nstep_impact = np.int64(t_impact / dt_orbit)
    #prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
    prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
    impact_orbit = prog_orbit[nstep_impact:]
    impact_orbit = impact_orbit[::-1]
    prog_orbit = prog_orbit[::-1]
    
    t_disrupt = -300*u.Myr
    minit = 7e4
    mfin = 1e3
    nrelease = 1
    n_times = (prog_orbit.t < t_disrupt).sum()
    prog_mass = np.linspace(minit, mfin, n_times)
    prog_mass = np.concatenate((prog_mass, np.zeros(len(prog_orbit.t) - n_times))) * u.Msun
    plt.plot(prog_orbit.t, prog_mass)
    model_present = mockstream.dissolved_fardal_stream(ham, prog_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)
    
    n_steps_disrupt = int(abs(t_disrupt / (prog_orbit.t[1]-prog_orbit.t[0])))
    model_present = model_present[:-2*n_steps_disrupt]
    
    model_gd1 = model_present.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    ind_gap = np.where((model_gd1.phi1.wrap_at(wangle)>-43*u.deg) & (model_gd1.phi1.wrap_at(wangle)<-33*u.deg))[0]

    n_times = (impact_orbit.t < t_disrupt).sum()
    prog_mass = np.linspace(minit, mfin, n_times)
    prog_mass = np.concatenate((prog_mass, np.zeros(len(impact_orbit.t) - n_times))) * u.Msun
    model = mockstream.dissolved_fardal_stream(ham, impact_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)

    n_steps_disrupt = int(abs(t_disrupt / (impact_orbit.t[1]-impact_orbit.t[0])))
    model = model[:-2*n_steps_disrupt]
    
    Nstar = np.shape(model)[0]
    ivalid = ind_gap < Nstar
    ind_gap = ind_gap[ivalid]
    
    xgap = np.median(model.xyz[:,ind_gap], axis=1)
    vgap = np.median(model.v_xyz[:,ind_gap], axis=1)
    
    
    ########################
    # Perturber at encounter
    
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
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    dt = 0.05*u.Myr
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # generate stream model
    potential_perturb = 2
    par_perturb = np.array([M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    wangle = 180*u.deg
    outdict = {'cg': cg}
    #pickle.dump(outdict, open('../data/fiducial.pkl', 'wb'))
    
    # load data
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))

    plt.close()
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0)

    plt.sca(ax[0])
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, rasterized=True)
    #plt.plot(g['phi1'], g['phi2'], 'ko', ms=2, alpha=0.7, mec='none')
    #plt.gca().set_aspect('equal')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.text(0.03, 0.9, 'Observed GD-1 stream', fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
    txt = plt.text(0.04, 0.75, 'Gaia proper motions\nPanSTARRS photometry', transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
    
    
    plt.sca(ax[1])
    #plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'k.', ms=3, alpha=0.2, rasterized=True)
    plt.plot(model_gd1.phi1.wrap_at(wangle), model_gd1.phi2, 'k.', ms=3, alpha=0.2, rasterized=True)
    #plt.plot(model_gd1.phi1.wrap_at(wangle)[Nstar:], model_gd1.phi2[Nstar:], 'k.', ms=1)

    plt.xlim(-70, -10)
    plt.ylim(-6,6)
    plt.yticks([-5,0,5])
    #plt.gca().set_aspect('equal')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.text(0.03, 0.9, 'Model of a perturbed GD-1', fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
    txt = plt.text(0.04, 0.75, """t = {:.0f} Myr
M = {:.0f}$\cdot$10$^6$ M$_\odot$
$r_s$ = {:.0f} pc
b = {:.0f} pc
V = {:.0f} km s$^{{-1}}$""".format(t_impact.to(u.Myr).value, M.to(u.Msun).value*1e-6, rs.to(u.pc).value, bnorm.to(u.pc).value, vnorm.to(u.km/u.s).value), transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
    
    #plt.close()
    #plt.figure()

    ##plt.plot(model.cylindrical.rho, model.z, 'k.', alpha=0.5, ms=0.2)
    ##plt.plot(model[ind_gap].cylindrical.rho, model[ind_gap].z, 'ro', alpha=0.5, ms=0.2)
    ##plt.plot(np.sqrt(xsub[0]**2 + xsub[1]**2).to(u.kpc), xsub[2].to(u.kpc), 'ro')
    ##plt.plot(np.sqrt(xgap[0]**2 + xgap[1]**2).to(u.kpc), xgap[2].to(u.kpc), 'ko')

    #plt.plot(model_present.cylindrical.rho, model_present.z, 'k.', alpha=0.5, ms=0.2)
    #plt.plot(model_present[ind_gap].cylindrical.rho, model_present.z[ind_gap], 'ro', alpha=0.5, ms=0.2)
    #plt.plot(model_present.cylindrical.rho[:100], model_present.z[:100], 'o')
    #plt.plot(model_present.cylindrical.rho[-100:], model_present.z[-100:], 'o')
    
    #plt.xlabel('R [kpc]')
    #plt.ylabel('z [kpc]')
    
    plt.tight_layout(h_pad=0)
    #plt.savefig('../plots/stream_encounter.png', dpi=200)
    #plt.savefig('../paper/stream_encounter.pdf', dpi=200)

####################
# Plot paper figures

def fiducial_params():
    """Return fiducial parameters"""
    
    bnorm = 10*u.pc
    bx =10*u.pc
    vnorm = 300*u.km/u.s
    vx = -300*u.km/u.s
    M = 6e6*u.Msun
    t_impact = 0.5*u.Gyr
    rs = 12*u.pc
    
    bnorm = 10*u.pc
    bx = 5*u.pc
    vnorm = 300*u.km/u.s
    vx = -300*u.km/u.s
    M = 4e6*u.Msun
    t_impact = 0.53*u.Gyr
    rs = 10*u.pc
    
    return (t_impact, M, rs, bnorm, bx, vnorm, vx)

def streak2orbit_fiducial():
    """"""
    
    phi = -20
    theta = 170
    bnorm = 15*u.pc
    bx = bnorm * np.cos(np.radians(phi))
    by = bnorm * np.sin(np.radians(phi))
    vnorm = 250*u.km/u.s
    vx = vnorm * np.cos(np.radians(theta))
    vy = vnorm * np.sin(np.radians(theta))
    M = 5e6*u.Msun
    t_impact = 0.495*u.Gyr
    rs = 0.1*rs_diemer(M)
    Tgap = 29.176*u.Myr
    
    return (t_impact, bx, by, vx, vy, M, rs, Tgap)

def generate_fiducial():
    """"""
    
    t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    
    cg, e = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t_impact, N=2000, fname='gd1_manfid', point_mass=False, verbose=False, model_return=True, fig_plot=False)
    
    outdict = {'cg': cg, 'e': e}
    pickle.dump(outdict, open('../data/fiducial.pkl', 'wb'))

def generate_streak2orbit_fiducial():
    """"""
    
    params_list = streak2orbit_fiducial()
    params_units = [p_.unit for p_ in params_list]
    x = [p_.value for p_ in params_list]
    x[5] = np.log10(x[5])
    
    lnprob_args = get_lnprobargs()
    res = lnprob_detailed(x, *lnprob_args)

    cg = res['stream']
    outdict = {'cg': cg}
    pickle.dump(outdict, open('../data/fiducial.pkl', 'wb'))

def plot_fiducial(generate=False):
    """Figure 1"""
    
    # params
    t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    wangle = 180*u.deg
    
    # load model
    if generate:
        generate_fiducial()
    pkl = pickle.load(open('../data/fiducial.pkl', 'rb'))
    cg = pkl['cg']
    
    # gap profile
    bins = np.linspace(-60,-20,30)
    bc = 0.5 * (bins[1:] + bins[:-1])
    db = bins[1] - bins[0]
    Nb = np.size(bc)
    f_gap = 0.5
    delta_phi2 = 0.5
    
    gap = np.load('../data/gap_properties.npz')
    phi1_edges = gap['phi1_edges']
    gap_position = gap['position']
    gap_width = gap['width']
    gap_yerr = gap['yerr']
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    
    p = np.load('../data/polytrack.npy')
    poly = np.poly1d(p)
    x_ = np.linspace(-100,0,100)
    y_ = poly(x_)
        
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(wangle).value))<delta_phi2
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(wangle).value, bins=bins)
    yerr = np.sqrt(h_model+1)/db
    h_model = h_model/db
    
    model_hat = np.median(h_model[hat_mask])
    h_model = h_model - model_hat
    
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    model_hat = np.minimum(model_hat, model_base*f_gap)
    ytop_model = tophat(bc, model_base, model_hat,  gap_position, gap_width)
    
    # rescale
    h_model = h_model / model_base
    ytop_model = ytop_model / model_base
    yerr = yerr / model_base
    
    # spur
    sp = np.load('../data/spur_track.npz')
    spx = sp['x']
    spy = sp['y']
    f = scipy.interpolate.interp1d(spx, spy, kind='quadratic')
    fx = np.linspace(-50, -30, 100)
    
    # observations
    g = Table.read('../data/members.fits')
    
    phi2_mask_data = np.abs(g['phi2'] - poly(g['phi1']))<delta_phi2
    h_data, be = np.histogram(g['phi1'][phi2_mask_data], bins=bins)
    yerr_data = np.sqrt(h_data+1)/db
    h_data = h_data/db
    
    data_hat = np.median(h_data[hat_mask])
    h_data = h_data - data_hat
    
    data_hat = np.median(h_data[hat_mask])
    data_base = np.median(h_data[base_mask])
    ytop_data = tophat(bc, data_base, data_hat, gap_position, gap_width)
    
    # rescale
    h_data = h_data / data_base
    ytop_data = ytop_data / data_base
    yerr_data = yerr_data / data_base
    
    plt.close()
    fig, ax = plt.subplots(2, 2, figsize=(13, 4.9), sharex='col', sharey='col', gridspec_kw = {'width_ratios':[3, 1]})

    plt.sca(ax[0][0])
    plt.plot(g['phi1'], g['phi2'], 'ko', ms=2, alpha=0.7, mec='none')
    plt.plot(fx, f(fx), '-', color='tomato', lw=3, alpha=0.6, zorder=0)
    
    plt.text(0.03, 0.9, 'Observed GD-1 stream', fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
    txt = plt.text(0.04, 0.75, 'Gaia proper motions\nPanSTARRS photometry', transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
    
    plt.ylabel('$\phi_2$ [deg]')
    plt.xlim(-80,0)
    plt.ylim(-10,9)
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[1][0])
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'ko', ms=4, alpha=0.7, mec='none')
    plt.plot(fx, f(fx), '-', color='tomato', lw=3, alpha=0.6)

    plt.text(0.03, 0.9, 'Model of a perturbed GD-1', fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
    txt = plt.text(0.04, 0.75, """t = {:.0f} Myr
M = {:.0f}$\cdot$10$^6$ M$_\odot$
$r_s$ = {:.0f} pc
b = {:.0f} pc
V = {:.0f} km s$^{{-1}}$""".format(t_impact.to(u.Myr).value, M.to(u.Msun).value*1e-6, rs.to(u.pc).value, bnorm.to(u.pc).value, vnorm.to(u.km/u.s).value), transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))

    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.xlim(-80,0)
    plt.ylim(-10,9)
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[0][1])
    plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='', alpha=0.7, lw=1.5)
    plt.plot(bc, h_data, 'wo', ms=6, mec='none')
    plt.plot(bc, h_data, 'ko', ms=6, mec='none', alpha=0.7)
    plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='')
    plt.plot(bc, ytop_data, '-', color='tomato', lw=3, alpha=0.6, zorder=0)
    
    plt.ylabel('$\\tilde{N}$ [deg$^{-1}$]')
    
    plt.sca(ax[1][1])
    plt.errorbar(bc, h_model, yerr=yerr, fmt='none', color='k', label='', alpha=0.7, lw=1.5)
    plt.plot(bc, h_model, 'wo', ms=6, mec='none')
    plt.plot(bc, h_model, 'ko', ms=6, mec='none', alpha=0.7)
    plt.plot(bc, ytop_model, '-', color='tomato', lw=3, alpha=0.6, zorder=0)
    
    plt.ylabel('$\\tilde{N}$ [deg$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    plt.xlim(-52, -28)
    plt.ylim(-0.9, 2.5)
    
    plt.tight_layout(h_pad=0.05)
    plt.savefig('../paper/data_model_comparison.pdf')
    
def fiducial_orbit():
    """"""
    x = np.array([-18.34075245, 3.02965498, 11.43273191])*u.kpc
    v = np.array([68.31170219, -156.98475247, 246.3519816])*u.km/u.s
    
    ir = x.value / np.linalg.norm(x.value)
    vt = np.cross(ir, v)*v.unit
    vr = v - vt
    print(vr, vt)
    print(np.linalg.norm(vr), np.linalg.norm(vt))

def generate_excursions():
    """Generate models by varying one parameter at a time"""
    
    # params
    t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    wangle = 180*u.deg
    N = 2000
    
    times = np.array([25,150,530,700,1000]) * u.Myr
    for e, t in enumerate(times):
        cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        out = {'cg': cg, 'e': e}
        pickle.dump(out, open('../data/ex3_t{:d}.pkl'.format(e), 'wb'))
    
    masses = np.array([1,2,4,7,10])*1e6 * u.Msun
    for e, m in enumerate(masses):
        cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=m, rs=rs, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        out = {'cg': cg, 'e': e}
        pickle.dump(out, open('../data/ex3_m{:d}.pkl'.format(e), 'wb'))
    
    bees = np.array([1,5,10,20,50]) * u.pc
    for e, b in enumerate(bees):
        cg, en = encounter(bnorm=b, bx=0.5*b, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        out = {'cg': cg, 'e': e}
        pickle.dump(out, open('../data/ex3_b{:d}.pkl'.format(e), 'wb'))
    
    vees = np.array([50,150,300,500,800]) * u.km/u.s
    for e, v in enumerate(vees):
        cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=v, vx=-v, M=M, rs=rs, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        out = {'cg': cg, 'e': e}
        pickle.dump(out, open('../data/ex3_v{:d}.pkl'.format(e), 'wb'))
    
    rses = np.array([0, 5, 10, 30, 100]) * u.pc
    for e, r in enumerate(rses):
        cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=r, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        out = {'cg': cg, 'e': e}
        pickle.dump(out, open('../data/ex3_r{:d}.pkl'.format(e), 'wb'))

def generate_streak2orbit_excursions():
    """Generate models by varying one parameter at a time"""
    
    # params
    phi = -20
    theta = 170
    params_list = streak2orbit_fiducial()
    params_units = [p_.unit for p_ in params_list]
    x0 = [p_.value for p_ in params_list]
    x0[5] = np.log10(x0[5])
    
    wangle = 180*u.deg
    N = 2000
    #print(params_list)
    
    times = np.array([25,150,495,700,1200]) * u.Myr
    for e, t in enumerate(times):
        x = x0[:]
        x[0] = t.to(params_units[0]).value
        lnprob_args = get_lnprobargs()

        res = lnprob_detailed(x, *lnprob_args)
        cg = res['stream']
        out = {'cg': cg}
        pickle.dump(out, open('../data/ex_t{:d}.pkl'.format(e), 'wb'))
    
    masses = np.array([1,2,5,7,10])*1e6 * u.Msun
    for e, m in enumerate(masses):
        x = x0[:]
        x[5] = np.log10(m.to(params_units[5]).value)
        lnprob_args = get_lnprobargs()

        res = lnprob_detailed(x, *lnprob_args)
        cg = res['stream']
        out = {'cg': cg}
        pickle.dump(out, open('../data/ex_m{:d}.pkl'.format(e), 'wb'))

    bees = np.array([1,5,15,25,50]) * u.pc
    for e, b in enumerate(bees):
        x = x0[:]
        x[1] = (b * np.cos(np.radians(phi))).to(params_units[1]).value
        x[2] = (b * np.sin(np.radians(phi))).to(params_units[2]).value
        lnprob_args = get_lnprobargs()

        res = lnprob_detailed(x, *lnprob_args)
        cg = res['stream']
        out = {'cg': cg}
        pickle.dump(out, open('../data/ex_b{:d}.pkl'.format(e), 'wb'))
    
    vees = np.array([30,150,250,500,800]) * u.km/u.s
    for e, v in enumerate(vees):
        x = x0[:]
        x[3] = (v * np.cos(np.radians(theta))).to(params_units[3]).value
        x[4] = (v * np.sin(np.radians(theta))).to(params_units[4]).value
        lnprob_args = get_lnprobargs()

        res = lnprob_detailed(x, *lnprob_args)
        cg = res['stream']
        out = {'cg': cg}
        pickle.dump(out, open('../data/ex_v{:d}.pkl'.format(e), 'wb'))

    rses = np.array([0, 5, 10, 30, 100]) * u.pc
    for e, r in enumerate(rses):
        x = x0[:]
        x[6] = r.to(params_units[6]).value
        lnprob_args = get_lnprobargs()

        res = lnprob_detailed(x, *lnprob_args)
        cg = res['stream']
        out = {'cg': cg}
        pickle.dump(out, open('../data/ex_r{:d}.pkl'.format(e), 'wb'))

def fiducial_excursions():
    """Loop appearance under different impact parameters"""
    
    # fiducial impact parameters
    t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    
    # visualization parameters
    N = 1000
    wangle = 180*u.deg
    color = '0.3'
    ms = 8
    lw = 3.5
    alpha = 0.7
    
    plt.close()
    fig, ax = plt.subplots(5,5,figsize=(12,12), sharex=True, sharey=True)
    
    for i in range(5):
        plt.sca(ax[4][i])
        plt.xlabel('$\phi_1$ [deg]')
        
        plt.xlim(-58,-22)
        plt.ylim(-6,6)

        plt.sca(ax[i][0])
        plt.ylabel('$\phi_2$ [deg]')
        
        plt.sca(ax[2][i])
        plt.text(0.9,0.2, 'fiducial', transform=plt.gca().transAxes, fontsize='small', va='center', ha='right')
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(2.3)
            plt.gca().spines[axis].set_edgecolor('0.2')
            
        plt.sca(ax[3][i])
        plt.gca().spines['top'].set_linewidth(2.3)
        plt.gca().spines['top'].set_edgecolor('0.2')
    
    #times = np.array([25,150,530,700,1000]) * u.Myr
    times = np.array([25,150,495,700,1200]) * u.Myr
    for e, t in enumerate(times):
        pkl = pickle.load(open('../data/ex_t{:d}.pkl'.format(e), 'rb'))
        cg = pkl['cg']
        #cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        
        color = mpl.cm.Purples(1 - np.abs(e-2)/4)
        
        plt.sca(ax[e][0])
        plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, '-', color=color, ms=ms, lw=lw, alpha=alpha)
        
        plt.text(0.1,0.8, 'T = {:.0f} Myr'.format(t.value), transform=plt.gca().transAxes, fontsize='small', va='center', ha='left')
    
    #masses = np.array([1,2,4,7,10])*1e6 * u.Msun
    masses = np.array([1,2,5,7,10])*1e6 * u.Msun
    for e, m in enumerate(masses):
        pkl = pickle.load(open('../data/ex_m{:d}.pkl'.format(e), 'rb'))
        cg = pkl['cg']
        #cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=m, rs=rs, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        
        color = mpl.cm.Blues(1 - np.abs(e-2)/4)

        plt.sca(ax[e][1])
        plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, '-', color=color, ms=ms, lw=lw, alpha=alpha)
        
        if m<1e7*u.Msun:
            plt.text(0.1,0.8, 'M = {:.0f}$\cdot10^6$ M$_\odot$'.format(m.value*1e-6), transform=plt.gca().transAxes, fontsize='small', va='center', ha='left')
        else:
            plt.text(0.1,0.8, 'M = {:.1f}$\cdot10^7$ M$_\odot$'.format(m.value*1e-7), transform=plt.gca().transAxes, fontsize='small', va='center', ha='left')
    
    #bees = np.array([1,5,10,20,50]) * u.pc
    bees = np.array([1,5,15,25,50]) * u.pc
    for e, b in enumerate(bees):
        pkl = pickle.load(open('../data/ex_b{:d}.pkl'.format(e), 'rb'))
        cg = pkl['cg']
        #cg, en = encounter(bnorm=b, bx=b, vnorm=vnorm, vx=vx, M=M, rs=rs, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        
        color = mpl.cm.Greens(1 - np.abs(e-2)/4)

        plt.sca(ax[e][2])
        plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, '-', color=color, ms=ms, lw=lw, alpha=alpha)
        
        plt.text(0.1,0.8, 'b = {:.0f} pc'.format(b.value), transform=plt.gca().transAxes, fontsize='small', va='center', ha='left')
    
    #vees = np.array([50,150,300,500,800]) * u.km/u.s
    vees = np.array([30,150,250,500,800]) * u.km/u.s
    for e, v in enumerate(vees):
        pkl = pickle.load(open('../data/ex_v{:d}.pkl'.format(e), 'rb'))
        cg = pkl['cg']
        #cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=v, vx=-v, M=M, rs=rs, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        
        color = mpl.cm.Oranges(1 - np.abs(e-2)/4)
        
        plt.sca(ax[e][3])
        plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, '-', color=color, ms=ms, lw=lw, alpha=alpha)
        
        plt.text(0.1,0.8, 'V = {:.0f} km s$^{{-1}}$'.format(v.value), transform=plt.gca().transAxes, fontsize='small', va='center', ha='left')
    
    rses = np.array([0, 5, 10, 30, 100]) * u.pc
    for e, r in enumerate(rses):
        pkl = pickle.load(open('../data/ex_r{:d}.pkl'.format(e), 'rb'))
        cg = pkl['cg']
        #cg, en = encounter(bnorm=bnorm, bx=bx, vnorm=vnorm, vx=vx, M=M, rs=r, t_impact=t_impact, N=N, point_mass=False, verbose=False, model_return=True, fig_plot=False)
        
        color = mpl.cm.Reds(0.7 - np.abs(e-2)/5)

        plt.sca(ax[e][4])
        plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, '-', color=color, ms=ms, lw=lw, alpha=alpha)
        
        plt.text(0.1,0.8, '$r_s$ = {:.0f} pc'.format(r.value), transform=plt.gca().transAxes, fontsize='small', va='center', ha='left')

    plt.tight_layout(h_pad=-0.1, w_pad=0.4)
    plt.savefig('../paper/excursions.pdf')
    plt.savefig('../plots/excursions.png', dpi=150)

def fancy_corner(label='', full=False, nstart=2000):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = sampler['chain']
    nwalkers = sampler['nwalkers']
    ntot, npar = np.shape(chain)
    #nstep = int(ntot/nwalkers)
    chain = trim_chain(chain, nwalkers, nstart, npar)
    
    abr = chain[:,:-3]
    abr[:,1] = np.sqrt(chain[:,1]**2 + chain[:,2]**2)
    abr[:,2] = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
    abr[:,0] = chain[:,0]
    abr[:,3] = np.log10(chain[:,6])
    abr[:,4] = chain[:,5]
    params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', 'log $r_s$/pc', 'log M/M$_\odot$']
    ind = (abr[:,3]>0) | (abr[:,2]>250)
    abr = abr[ind]
    chain = abr
    npar = np.shape(chain)[1]
    
    lims = [[0.,1], [0.1,30], [200,500], [-1,2], [5.8,8]]
    t_impact = 0.495*u.Gyr
    M = 5e6*u.Msun
    rs = 0.1*rs_diemer(M)
    bnorm = 15*u.pc
    vnorm = 250*u.km/u.s
    
    pfid = [t_impact.to(u.Gyr).value, bnorm.to(u.pc).value, vnorm.to(u.km/u.s).value, np.log10(rs.to(u.pc).value), np.log10(M.to(u.Msun).value)]
    
    
    plt.close()
    fig = corner.corner(chain, bins=50, labels=params, plot_datapoints=False, range=lims, smooth=2, smooth1d=2, color='0.1')
    ax = fig.get_axes()
    
    for i in range(npar-1):
        for j in range(i+1, npar):
            ind = i + (j-1)*npar + npar
            plt.sca(ax[ind])
            plt.plot(pfid[i], pfid[j], '*', ms=20, mec='orangered', mew=1.5, color='orange')

    plt.savefig('../paper/corner.pdf')
    plt.savefig('../plots/fancy_corner.png')


