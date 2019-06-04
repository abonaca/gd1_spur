from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
#from astropy.constants import G, c as c_
#from astropy.io import fits

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.dynamics import mockstream

from gd1 import *

def impact_params():
    """"""
    t_impact = 0.495*u.Gyr
    M = 5e6*u.Msun
    rs = 0.1*rs_diemer(M)
    bnorm = 15*u.pc
    bx = 6*u.pc
    vnorm = 250*u.km/u.s
    vx = -25*u.km/u.s
    
    return (t_impact, M, rs, bnorm, bx, vnorm, vx)

def perturber_orbit():
    """"""
    pkl = pickle.load(open('../data/fiducial_at_encounter.pkl', 'rb'))
    
    xsub = pkl['xsub']
    vsub = pkl['vsub']
    c = coord.Galactocentric(x=xsub[0], y=xsub[1], z=xsub[2], v_x=vsub[0], v_y=vsub[1], v_z=vsub[2])
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    dt_orbit = 0.5*u.Myr
    orbit_rr = ham.integrate_orbit(w0, dt=-dt_orbit, t2=-0.5*495*u.Myr, t1=0*u.Myr)
    winit = orbit_rr.w()[:,-1]
    
    orbit = ham.integrate_orbit(winit, dt=dt_orbit, t1=0*u.Myr, t2=1.5*495*u.Myr)
    
    return orbit

def get_orbit():
    """"""
    t_impact, M, rs, bnorm, bx, vnorm, vx = impact_params()

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
    prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
    #impact_orbit = prog_orbit[nstep_impact:]
    #impact_orbit = impact_orbit[::-1]
    prog_orbit = prog_orbit[::-1]
    
    #print(nstep_impact, impact_orbit)
    
    return prog_orbit

def fiducial_orbit():
    """"""
    
    np.random.seed(143531)
    
    ##t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    #t_impact = 0.495*u.Gyr
    #M = 5e6*u.Msun
    #rs = 0.1*rs_diemer(M)
    #bnorm = 15*u.pc
    #bx = 6*u.pc
    #vnorm = 250*u.km/u.s
    #vx = -25*u.km/u.s

    ## load one orbital point
    #pos = np.load('../data/log_orbit.npy')
    #phi1, phi2, d, pm1, pm2, vr = pos

    #c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    #w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    ## best-fitting orbit
    #dt = 0.5*u.Myr
    #n_steps = 120
    #wangle = 180*u.deg

    ## integrate back in time
    #fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    
    #prog_phi0 = -20*u.deg

    #model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    #prog_i = np.abs(model_gd1.phi1.wrap_at(180*u.deg) - prog_phi0).argmin()
    #prog_w0 = fit_orbit[prog_i]
    
    #dt_orbit = 0.5*u.Myr
    #nstep_impact = np.int64(t_impact / dt_orbit)
    #prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
    #impact_orbit = prog_orbit[nstep_impact:]
    #impact_orbit = impact_orbit[::-1]
    #prog_orbit = prog_orbit[::-1]
    
    prog_orbit = get_orbit()

    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(10,10), sharex='col', sharey='row')
    
    plt.sca(ax[0][0])
    plt.plot(prog_orbit.x, prog_orbit.y, 'k-')
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[1][0])
    plt.plot(prog_orbit.x, prog_orbit.z, 'k-')
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[1][1])
    plt.plot(prog_orbit.y, prog_orbit.z, 'k-')
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[0][1])
    plt.axis('off')
    
    plt.tight_layout()

def orbit_equatorial():
    """"""
    
    prog_orbit = get_orbit()
    c = coord.Galactocentric(x=prog_orbit.x, y=prog_orbit.y, z=prog_orbit.z)
    ceq = c.transform_to(coord.ICRS)
    
    plt.close()
    plt.figure()
    
    plt.plot(ceq.ra, ceq.dec, 'k-')
    
    plt.tight_layout()

def plain_model():
    """Unperturbed model of GD-1"""
    
    np.random.seed(143531)
    
    t_impact, M, rs, bnorm, bx, vnorm, vx = impact_params()
    dt_orbit = 0.5*u.Myr
    nstep_impact = np.int64(t_impact / dt_orbit)
    
    prog_orbit = get_orbit()
    
    wangle = 180*u.deg
    t_disrupt = -300*u.Myr
    minit = 7e4
    mfin = 1e3
    nrelease = 1
    n_times = (prog_orbit.t < t_disrupt).sum()
    prog_mass = np.linspace(minit, mfin, n_times)
    prog_mass = np.concatenate((prog_mass, np.zeros(len(prog_orbit.t) - n_times))) * u.Msun
    
    # stream model at the present
    model = mockstream.dissolved_fardal_stream(ham, prog_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease, snapshot_filename='../data/model_unperturbed.h5')
    
    # angular coordinates
    r = np.sqrt(model.x**2 + model.y**2 + model.z**2)
    theta = np.arccos(model.z/r)
    phi = coord.Angle(np.arctan2(model.y, model.x)).wrap_at(0*u.deg)
    
    # sky coordinates
    c = coord.Galactocentric(x=model.x, y=model.y, z=model.z)
    ceq = c.transform_to(coord.ICRS)
    cgd = c.transform_to(gc.GD1)

    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(8,8))
    
    plt.sca(ax[0])
    plt.plot(model.y, model.z, 'k.', ms=1)
    plt.plot(model.y[::2], model.z[::2], 'r.', ms=1)
    
    plt.sca(ax[1])
    #plt.plot(phi, theta, 'k.', ms=1)
    plt.plot(ceq.ra, ceq.dec, 'k.', ms=1)
    plt.plot(ceq.ra[::2], ceq.dec[::2], 'r.', ms=1)
    
    plt.sca(ax[2])
    plt.plot(cgd.phi1.wrap_at(180*u.deg), cgd.phi2, 'k.', ms=1)
    plt.plot(cgd.phi1[::2].wrap_at(180*u.deg), cgd.phi2[::2], 'r.', ms=1)
    
    plt.tight_layout()

def fiducial_model():
    """"""
    np.random.seed(143531)
    
    t_impact, M, rs, bnorm, bx, vnorm, vx = impact_params()
    dt_orbit = 0.5*u.Myr
    nstep_impact = np.int64((t_impact / dt_orbit).decompose())
    
    prog_orbit = get_orbit()
    
    wangle = 180*u.deg
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

    prog_orbit = prog_orbit[::-1]
    impact_orbit = prog_orbit[nstep_impact:]
    impact_orbit = impact_orbit[::-1]

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
    #potential_perturb = 2
    #par_perturb = np.array([0*M.si.value, rs.si.value, 0, 0, 0])
    
    #x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    #stream = {}
    #stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    #stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    #c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    #cg = c.transform_to(gc.GD1)
    #wangle = 180*u.deg
    #outdict = {'cg': cg}
    #pickle.dump(outdict, open('../data/fiducial_noperturb.pkl', 'wb'))
    
    # generate perturbed stream model
    potential_perturb = 2
    par_perturb = np.array([M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    wangle = 180*u.deg
    #outdict = {'cg': cg}
    #pickle.dump(outdict, open('../data/fiducial_perturb.pkl', 'wb'))
    
    #plt.close()
    #plt.plot(stream['x'][1], stream['x'][2], 'k.')
    
    plt.close()
    plt.figure(figsize=(10,5))
    plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.', ms=1)
    plt.xlim(-80,0)
    plt.ylim(-10,10)
    plt.tight_layout()

def read_fiducial():
    """"""
    pkl = pickle.load(open('../data/fiducial_perturb_python3.pkl', 'rb'))
    
    return pkl['cg']


# gd-1 formation

def plot_fullorbit():
    """"""
    
    orbit = get_orbit()
    ind = orbit.t<-2.2*u.Gyr
    orbit = orbit[ind]
    
    t_disrupt = -300*u.Myr
    minit = 7e4
    mfin = 1e3
    nrelease = 1
    prog_orbit = orbit
    
    n_times = (prog_orbit.t < t_disrupt).sum()
    prog_mass = np.linspace(minit, mfin, n_times)
    prog_mass = np.concatenate((prog_mass, np.zeros(len(prog_orbit.t) - n_times))) * u.Msun
    model = mockstream.dissolved_fardal_stream(ham, prog_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)
    
    # sky coordinates
    c = coord.Galactocentric(x=model.x, y=model.y, z=model.z)
    
    plt.close()
    fig = plt.figure(figsize=(16,9))
    fig.add_subplot(111, position=[0,0,1,1])
    #plt.plot(orbit.y[0], orbit.z[0], 'ro')
    pdisk = mpl.patches.Ellipse((0,0), 30, 0.3, color='orange')
    patch = plt.gca().add_patch(pdisk)
    plt.plot(-8.3, 0, '*', color='darkorange', ms=30)
    
    plt.plot(orbit.x[-1], orbit.z[-1], 'ko', ms=10)
    plt.plot(orbit.x, orbit.z, 'k-', alpha=0.3)
    plt.plot(c.x, c.z, 'k.', ms=4, alpha=0.1)
    
    plt.text(0.05, 0.9, '{:4.0f} million years'.format((orbit.t[-1]-orbit.t[0]).to(u.Myr).value), transform=plt.gca().transAxes, fontsize=20)
    
    dx = 30
    dy = dx*9/16
    plt.xlim(-dx, dx)
    plt.ylim(-dy, dy)
    
    plt.gca().axis('off')
    plt.gca().tick_params(labelbottom=False, labelleft=False)
    
def generate_formation_snapshot(t):
    """"""
    
    orbit = get_orbit()
    ind = orbit.t<=t
    orbit = orbit[ind]
    
    np.random.seed(49382)
    t_disrupt = -300*u.Myr
    minit = 7e4
    mfin = 1e3
    nrelease = 1
    prog_orbit = orbit
    
    if np.size(orbit.t)>1:
        n_times = (prog_orbit.t < t_disrupt).sum()
        prog_mass = np.linspace(minit, mfin, n_times)
        prog_mass = np.concatenate((prog_mass, np.zeros(len(prog_orbit.t) - n_times))) * u.Msun
        
        model = mockstream.dissolved_fardal_stream(ham, prog_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)
        c = coord.Galactocentric(x=model.x, y=model.y, z=model.z)
    else:
        c = coord.Galactocentric(x=orbit.x, y=orbit.y, z=orbit.z)
    
    return (orbit, c)

def formation_series():
    """"""
    times = np.linspace(-3,-2.2,800)*u.Gyr
    
    for e, t in enumerate(times[:]):
        orb, c = generate_formation_snapshot(t)

        orbit = {}
        orbit['t'] = orb.t
        orbit['x'] = orb.x
        orbit['y'] = orb.y
        orbit['z'] = orb.z
        outdict = {'orbit': orbit, 'c': c}

        pickle.dump(outdict, open('../data/vis_formation/snap.{:03d}.pkl'.format(e), 'wb'))

def movplot_formation():
    """"""
    N = 800
    
    for e in range(N):
        pkl = pickle.load(open('../data/vis_formation/snap.{:03d}.pkl'.format(e), 'rb'))
        orbit = pkl['orbit']
        c = pkl['c']
        
        plt.close()
        fig = plt.figure(figsize=(16,9))
        fig.add_subplot(111, position=[0,0,1,1])
        #plt.plot(orbit.y[0], orbit.z[0], 'ro')
        pdisk = mpl.patches.Ellipse((0,0), 30, 0.3, color='orange')
        patch = plt.gca().add_patch(pdisk)
        plt.plot(-8.3, 0, '*', color='darkorange', ms=30)
        
        plt.plot(orbit['x'][-1], orbit['z'][-1], 'ko', ms=10)
        plt.plot(orbit['x'], orbit['z'], 'k-', alpha=0.3)
        plt.plot(c.x, c.z, 'k.', ms=4, alpha=0.1)
        
        plt.text(0.05, 0.9, '{:4.0f} million years'.format((orbit['t'][-1]-orbit['t'][0]).to(u.Myr).value), transform=plt.gca().transAxes, fontsize=20)
        
        dx = 30
        dy = dx*9/16
        plt.xlim(-dx, dx)
        plt.ylim(-dy, dy)
        
        plt.gca().axis('off')
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.savefig('../plots/mov_formation/mov.{:03d}.png'.format(e))


# pre-encounter

def fiducial_at_start():
    """Create fiducial model at the beginning of the movie"""
    
    np.random.seed(143531)
    t_impact=2*0.495*u.Gyr

    
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
    
    xsub = np.ones(3)*u.kpc
    vsub = np.ones(3)*u.km/u.s
    
    outdict = {'model': model, 'xsub': xsub, 'vsub': vsub}
    pickle.dump(outdict, open('../data/fiducial_at_start.pkl', 'wb'))

def generate_start_time(t_impact=0.495*u.Gyr, graph=False):
    """Generate fiducial model at t_impact after the impact"""
    
    # impact parameters
    M = 0*5e6*u.Msun
    rs = 10*u.pc
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    dt = 0.05*u.Myr
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    pkl = pickle.load(open('../data/fiducial_at_start.pkl', 'rb'))
    model = pkl['model']
    xsub = pkl['xsub']
    vsub = pkl['vsub']
    
    # generate perturbed stream model
    potential_perturb = 2
    par_perturb = np.array([0*M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    wangle = 180*u.deg
    
    if graph:
        plt.close()
        plt.figure(figsize=(10,5))
        plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.', ms=1)
        plt.xlim(-80,0)
        plt.ylim(-10,10)
        plt.tight_layout()
    
    return cg

def start_time_series():
    """"""
    
    #times = np.linspace(0,495,400)*u.Myr
    times = np.linspace(-1.5*0.495,-0.495,200)*u.Gyr
    
    for e, t in enumerate(times[:]):
        #print(e)
        #cg = generate_start_time(t_impact=t)
        
        orb, cg = generate_formation_snapshot(t)
        cgal = cg.transform_to(coord.Galactocentric)
        outdict = {'cg': cg, 'cgal': cgal}
        pickle.dump(outdict, open('../data/vis_start/start_{:03d}.pkl'.format(e), 'wb'))

def movplot_start_old():
    """"""
    N = 400
    orbit = get_orbit()
    ind = orbit.t>-2*0.495*u.Gyr
    orbit = orbit[ind]
    dt_step = 495*u.Myr/N
    dt_orbit = orbit.t[1] - orbit.t[0]

    w = 16
    h = 9
    
    for i in range(N):
        pkl = pickle.load(open('../data/fiducial_vis/start_{:03d}.pkl'.format(i), 'rb'))
        cgal = pkl['cgal']
        
        Nstar = np.size(cgal.x)
        icen = np.int64(i * dt_step / dt_orbit)
        
        plt.close()
        fig = plt.figure(figsize=(16,9))
        #fig.add_subplot(111, aspect='equal', adjustable='datalim', position=[0,0,1,1])
        fig.add_subplot(111, position=[0,0,1,1])
        
        plt.plot(cgal.y.to(u.kpc), cgal.z.to(u.kpc), 'k.', ms=2, alpha=0.2)
        #plt.plot(orbit.y[icen].to(u.kpc), orbit.z[icen].to(u.kpc), 'r.', ms=7)
        
        #x1_init, x2_init, y1_init, y2_init = -20, 20, -10, 12.5
        #x1_init, x2_init, y1_init, y2_init = -24, 24, -11, 16
        #plt.xlim(x1_init, x2_init)
        #plt.ylim(y1_init, y2_init)
        
        xc = orbit.y[icen].to(u.kpc).value
        yc = orbit.z[icen].to(u.kpc).value
        dw = 20 * (0.5 + 0.5*i/(N-1))
        dh = dw * h/w
        
        plt.xlim(xc-dw, xc+dw)
        plt.ylim(yc-dh, yc+dh)
        
        #print(xc-dw, xc+dw)
        #print(yc-dh, yc+dh)
        
        plt.gca().axis('off')
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
        #plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('{:0.0f}'))
        #plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('{:0.0f}'))
        plt.savefig('../plots/mov_start/mov.{:03d}.png'.format(i))
        i += 1

def movplot_start():
    """"""
    N = 200
    # stream step
    times = np.linspace(-1.5*0.495,-0.495,200)*u.Gyr
    dt_step = (times[1] - times[0]).to(u.Myr)
    
    # orbit step
    orbit = get_orbit()
    ind = orbit.t>=-1.5*0.495*u.Gyr
    orbit = orbit[ind]
    dt_orbit = orbit.t[1] - orbit.t[0]
    
    # progenitor orbit
    porbit = perturber_orbit()
    ind = porbit.t<=0.5*0.495*u.Gyr
    porbit = porbit[ind]
    dt_porbit = porbit.t[1] - porbit.t[0]

    w = 16
    h = 9
    dpi = 1920./16.-1
    
    for i in range(N):
        pkl = pickle.load(open('../data/vis_start/start_{:03d}.pkl'.format(i), 'rb'))
        cgal = pkl['cgal']
        
        Nstar = np.size(cgal.x)
        icen = np.int64(i * dt_step / dt_orbit)
        iper = np.int64(i * dt_step / dt_porbit)
        
        plt.close()
        fig = plt.figure(figsize=(16,9))
        #fig.add_subplot(111, aspect='equal', adjustable='datalim', position=[0,0,1,1])
        fig.add_subplot(111, position=[0,0,1,1])
        
        plt.plot(cgal.y.to(u.kpc), cgal.z.to(u.kpc), 'k.', ms=2, alpha=0.2)
        
        plt.plot(porbit.y[iper].to(u.kpc), porbit.z[iper].to(u.kpc), 'o', color='orange', mec='darkorange', mew=2, ms=16, zorder=0)
        #plt.plot(orbit.y[icen].to(u.kpc), orbit.z[icen].to(u.kpc), 'r.', ms=7)
        
        #x1_init, x2_init, y1_init, y2_init = -20, 20, -10, 12.5
        #x1_init, x2_init, y1_init, y2_init = -24, 24, -11, 16
        #plt.xlim(x1_init, x2_init)
        #plt.ylim(y1_init, y2_init)
        
        xc = orbit.y[icen].to(u.kpc).value
        yc = orbit.z[icen].to(u.kpc).value
        dw = 20 * (0.5 + 0.5*i/(N-1))
        dh = dw * h/w
        
        plt.xlim(xc-dw, xc+dw)
        plt.ylim(yc-dh, yc+dh)
        
        #print(xc-dw, xc+dw)
        #print(yc-dh, yc+dh)
        
        plt.gca().axis('off')
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
        #plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('{:0.0f}'))
        #plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('{:0.0f}'))
        plt.savefig('../plots/mov_start/mov.{:03d}.png'.format(i), dpi=dpi)
        i += 1


# post-encounter

def fiducial_at_encounter():
    """Create fiducial model at the time of encounter"""
    
    np.random.seed(143531)

    t_impact=0.495*u.Gyr
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
    
    outdict = {'model': model, 'xsub': xsub, 'vsub': vsub}
    pickle.dump(outdict, open('../data/fiducial_at_encounter.pkl', 'wb'))
    
def generate_encounter_time(t_impact=0.495*u.Gyr, graph=False):
    """Generate fiducial model at t_impact after the impact"""
    
    # impact parameters
    M = 5e6*u.Msun
    rs = 10*u.pc
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    dt = 0.05*u.Myr
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.to(u.m/u.s).value, q.value, rhalo.to(u.m).value])
    
    pkl = pickle.load(open('../data/fiducial_at_encounter.pkl', 'rb'))
    model = pkl['model']
    xsub = pkl['xsub']
    vsub = pkl['vsub']
    
    # generate perturbed stream model
    potential_perturb = 2
    par_perturb = np.array([M.to(u.kg).value, rs.to(u.m).value, 0, 0, 0])
    #print(vsub.si, par_perturb)
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.to(u.m).value, vsub.to(u.m/u.s).value, Tenc.to(u.s).value, t_impact.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, model.x.to(u.m).value, model.y.to(u.m).value, model.z.to(u.m).value, model.v_x.to(u.m/u.s).value, model.v_y.to(u.m/u.s).value, model.v_z.to(u.m/u.s).value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    wangle = 180*u.deg
    
    if graph:
        plt.close()
        plt.figure(figsize=(10,5))
        plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'k.', ms=1)
        plt.xlim(-80,0)
        plt.ylim(-10,10)
        plt.tight_layout()
    
    return cg

def encounter_time_series():
    """"""
    
    times = np.linspace(0,495,400)*u.Myr
    
    for e, t in enumerate(times):
        print(e)
        cg = generate_fiducial_time(t_impact=t)
        cgal = cg.transform_to(coord.Galactocentric)
        outdict = {'cg': cg, 'cgal': cgal}
        pickle.dump(outdict, open('../data/fiducial_vis/encounter_{:03d}.pkl'.format(e), 'wb'))

def thread_orbit():
    """"""
    
    orbit = get_orbit()
    ind = orbit.t>-1.5*0.495*u.Gyr
    orbit = orbit[ind]
    #print(orbit.t)
    #print(np.size(orbit.t))
    ihalf = orbit.t==-0.495*u.Gyr
    
    porbit = perturber_orbit()
    #cg = read_fiducial()
    #cgal = cg.transform_to(coord.Galactocentric)
    
    plt.close()
    fig = plt.figure(figsize=(16,9))
    fig.add_subplot(111, aspect='equal', adjustable='datalim')

    plt.plot(orbit.y, orbit.z, 'k-', alpha=0.3)
    plt.plot(orbit.y[0], orbit.z[0], 'ko')
    plt.plot(orbit.y[ihalf], orbit.z[ihalf], 'rx', ms=10)

    plt.plot(porbit.y, porbit.z, 'r-', alpha=0.3)
    plt.plot(porbit.y[0], porbit.z[0], 'ro', alpha=0.3)
    plt.plot(porbit.y[-1], porbit.z[-1], 'bo', alpha=0.3)
    plt.plot(porbit.y[494], porbit.z[494], 'bx', alpha=0.3)
    print(porbit.t[0], porbit.t[-1])
    #print(porbit.t)
    
    for e in range(0,400,100):
        pkl = pickle.load(open('../data/fiducial_vis/encounter_{:03d}.pkl'.format(e), 'rb'))
        cgal = pkl['cgal']
        plt.plot(cgal.y.to(u.kpc), cgal.z.to(u.kpc), '.', color=mpl.cm.viridis(e/400), ms=0.1)
    
    plt.tight_layout()

def movplot_encounter():
    """"""
    N = 400
    
    orbit = get_orbit()
    ind = orbit.t>-2*0.495*u.Gyr
    orbit = orbit[ind]
    dt_step = 495*u.Myr/N
    dt_orbit = orbit.t[1] - orbit.t[0]
    
    icen = np.int64((N-1) * dt_step / dt_orbit)
    xc = orbit.y[icen].to(u.kpc).value
    yc = orbit.z[icen].to(u.kpc).value
    dw = 20
    dh = dw * 9/16
    
    x1_init, x2_init, y1_init, y2_init = xc-dw, xc+dw, yc-dh, yc+dh
    x1_final, x2_final, y1_final, y2_final = -24, 24, -11, 16
    
    Nzoom = 100
    dpi = 1920./16.-1
    
    # progenitor orbit
    porbit = perturber_orbit()
    ind = porbit.t>=0.5*0.495*u.Gyr
    porbit = porbit[ind]
    dt_porbit = porbit.t[1] - porbit.t[0]
    
    for i in range(283,N):
        pkl = pickle.load(open('../data/fiducial_vis/encounter_{:03d}.pkl'.format(i), 'rb'))
        cgal = pkl['cgal']
        iper = np.int64(i * dt_step / dt_porbit)
        
        plt.close()
        fig = plt.figure(figsize=(16,9))
        #fig.add_subplot(111, aspect='equal', adjustable='datalim', position=[0,0,1,1])
        fig.add_subplot(111, position=[0,0,1,1])
        
        plt.plot(cgal.y.to(u.kpc), cgal.z.to(u.kpc), 'k.', ms=2, alpha=0.2)
        
        if i<130:
            plt.plot(porbit.y[iper].to(u.kpc), porbit.z[iper].to(u.kpc), 'o', color='orange', mec='darkorange', mew=2, ms=16, zorder=0)

        #x1_init, x2_init, y1_init, y2_init = -20, 20, -10, 12.5
        #x1_init, x2_init, y1_init, y2_init = -24, 24, -11, 16
        #plt.xlim(x1_init, x2_init)
        #plt.ylim(y1_init, y2_init)
        flin = i/Nzoom
        if i<Nzoom:
            plt.xlim((1-flin)*x1_init + flin*x1_final, (1-flin)*x2_init + flin*x2_final)
            plt.ylim((1-flin)*y1_init + flin*y1_final, (1-flin)*y2_init + flin*y2_final)
        else:
            plt.xlim(x1_final, x2_final)
            plt.ylim(y1_final, y2_final)
        
        plt.gca().axis('off')
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.savefig('../plots/mov_encounter/mov.{:03d}.png'.format(i), dpi=dpi)
        i += 1

# galactocentric to GD-1 coordinates

def compare_cartesian_sky():
    """"""
    galcen_distance = coord.Galactocentric.galcen_distance

    cg = read_fiducial()
    cgal = cg.transform_to(coord.Galactic)
    cgc = cg.transform_to(coord.Galactocentric)
    
    dist = np.median(cgal.distance)
    cgal_unidist = coord.Galactic(l=cgal.l, b=cgal.b, distance=dist)
    cgc_unidist = cgal_unidist.transform_to(coord.Galactocentric)
    
    #cgal_sphere = coord.Galactic(l=cgal.l, b=cgal.b, distance=1*u.kpc)
    #cgc_sphere = cgal_sphere.transform_to(coord.Galactocentric)
    #l = coord.Angle(np.arctan2(cgc_sphere.y.to(u.kpc), cgc_sphere.x.to(u.kpc)-galcen_distance))
    #b = coord.Angle(np.arctan2(cgc_sphere.z.to(u.kpc), np.sqrt((cgc_sphere.x.to(u.kpc)-galcen_distance)**2 + cgc_sphere.y.to(u.kpc)**2)))
    l = coord.Angle(np.arctan2(cgc.y.to(u.kpc), cgc.x.to(u.kpc)+galcen_distance))
    b = coord.Angle(np.arctan2(cgc.z.to(u.kpc), np.sqrt((cgc.x.to(u.kpc)+galcen_distance)**2 + cgc.y.to(u.kpc)**2)))
    
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10))
    
    plt.sca(ax[0])
    plt.plot(cgc.y, cgc.z, 'k.')
    plt.plot(cgc_unidist.y, cgc_unidist.z, 'r.')
    
    plt.sca(ax[1])
    plt.plot(cgal.l.rad, cgal.b.rad, 'k.')
    plt.plot(cgal_unidist.l.rad, cgal_unidist.b.rad, 'r.')
    #plt.plot(cgal_sphere.l.rad, cgal_sphere.b.rad, 'b.')
    #plt.plot(cgc_sphere.y, cgc_sphere.z, 'g.')
    plt.plot(l.wrap_at(360*u.deg), b, 'm.')
    
    plt.tight_layout()

def movplot_transition():
    """"""
    
    cg = read_fiducial()
    cgal = cg.transform_to(coord.Galactocentric)
    clb = cg.transform_to(coord.Galactic)

    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    
    N = 300
    i = 0
    fvec = np.logspace(-3,0,N-1)
    fvec = np.concatenate([np.array([0.]), fvec])
    dpi = 1920./16.-1

    #for i in range(N):
    #fvec = fvec[172:176]
    for f in fvec[:]:
        #flin = i/(N-1)
        flin = f
        
        # linear
        x = (1-f)*cgal.y.to(u.kpc).value + f*cg.phi1.wrap_at(180*u.deg).deg
        y = (1-f)*cgal.z.to(u.kpc).value + f*cg.phi2.deg
        
        ## tanh
        #x = np.tanh()*cgal.y.to(u.kpc).value + f*cg.phi1.wrap_at(180*u.deg).deg
        #y = (1-f)*cgal.z.to(u.kpc).value + f*cg.phi2.deg
        
        
        plt.close()
        fig = plt.figure(figsize=(16,9))
        #fig.add_subplot(111, aspect='equal', adjustable='datalim', position=[0,0,1,1])
        fig.add_subplot(111, position=[0,0,1,1])
        #fig.add_subplot(111, aspect='equal', adjustable='datalim')
        #plt.plot(x, y, '.', color=mpl.cm.viridis(f))
        plt.plot(x, y, 'k.', ms=3+f, alpha=0.2)
        
        x1_init, x2_init, y1_init, y2_init = -20, 20, -10, 12.5
        x1_init, x2_init, y1_init, y2_init = -24, 24, -11, 16
        x1_final, x2_final, y1_final, y2_final = -70, -10, -15, 30

        plt.xlim((1-flin)*x1_init + flin*x1_final, (1-flin)*x2_init + flin*x2_final)
        plt.ylim((1-flin)*y1_init + flin*y1_final, (1-flin)*y2_init + flin*y2_final)
        
        #print(i, f, np.min(x), np.max(x), np.min(y), np.max(y))
        #print(i, f, (1-flin)*x1_init + flin*x1_final, (1-flin)*x2_init + flin*x2_final, plt.gca().get_xlim())
        #print(i, f, (1-flin)*y1_init + flin*y1_final, (1-flin)*y2_init + flin*y2_final, plt.gca().get_ylim())
        
        plt.gca().axis('off')
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        #plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
        #plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
        #plt.gca().set_aspect('equal')
        #plt.tight_layout()
        plt.savefig('../plots/gal2sky/gal2gd1.{:03d}.png'.format(i), dpi=dpi)
        i += 1
    
    i = 300
    for e in range(0,60,1):
        alpha = e/59
        
        plt.close()
        fig = plt.figure(figsize=(16,9))
        fig.add_subplot(111, position=[0,0,1,1])
        plt.plot(x, y, 'k.', ms=3+f, alpha=0.2)

        plt.scatter(g['phi1'], g['phi2']+15, s=g['pmem']*4, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, zorder=0, alpha=alpha)
        
        plt.text(-64, 20, 'Observed GD-1 stellar stream', alpha=alpha, fontsize=25)
        plt.text(-64, 3, 'Model stellar stream', alpha=alpha, fontsize=25)
        
        plt.xlim(x1_final, x2_final)
        plt.ylim(y1_final, y2_final)
        
        plt.gca().axis('off')
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.savefig('../plots/gal2sky/gal2gd1.{:03d}.png'.format(e+i), dpi=dpi)


# blender inputs

def blender_snaps_encounter():
    """"""
    
    times = np.linspace(0,495,10)*u.Myr
    
    for e, t in enumerate(times):
        cg = generate_encounter_time(t_impact=t)
        cgal = cg.transform_to(coord.Galactocentric)
        print(e, t, np.size(cg.phi1))
        np.savez('../data/blender_vis/encounter_{:03d}'.format(e), x=cgal.x.to(u.kpc).value, y=cgal.y.to(u.kpc).value, z=cgal.z.to(u.kpc).value)

def test_blender_snaps():
    """"""
    
    d1 = np.load('../data/blender_vis/encounter_001.npz')
    
    plt.close()
    plt.figure(figsize=(16,9))
    
    for i in range(10):
        d0 = np.load('../data/blender_vis/encounter_{:03d}.npz'.format(i))
        plt.plot(d0['y'], d0['z'], '.')
    
    plt.xlim(-24,24)
    plt.ylim(-11,16)
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()


# smooth rotation

def gd1_pole():
    """"""
    
    c = coord.ICRS(ra=0*u.deg, dec=90*u.deg)
    cgd = c.transform_to(gc.GD1)
    
    print(cgd)
    
    cgd = gc.GD1(phi1=0*u.deg, phi2=90*u.deg)
    c = cgd.transform_to(coord.ICRS)
    
    print(c)
    
    c1 = [0*u.deg, 90*u.deg]
    c2 = [c.ra, c.dec]
    
    plt.close()
    plt.figure(figsize=(10,5))
    plt.subplot(111, projection='mollweide')
    plt.plot(c1[0].to(u.rad), c1[1].to(u.rad), 'ro', ms=10)
    plt.plot(c2[0].to(u.rad), c2[1].to(u.rad), 'ro', ms=10)
    
    for f in np.linspace(0,1,100):
        c3 = great_circle(c1, c2, f=f)
        plt.plot(c3[0].to(u.rad), c3[1].to(u.rad), 'o', color=mpl.cm.viridis(f), ms=4)
    
    plt.tight_layout()

def great_circle(c1, c2, f=0.5):
    """"""
    v1 = np.array([np.cos(c1[0])*np.cos(c1[1]), np.sin(c1[0])*np.cos(c1[1]), np.sin(c1[1])])
    v2 = np.array([np.cos(c2[0])*np.cos(c2[1]), np.sin(c2[0])*np.cos(c2[1]), np.sin(c2[1])])
    v3 = (1-f)*v1 + f*v2
    v3 = v3 / np.linalg.norm(v3)
    
    c3 = [np.arctan2(v3[1], v3[0])*u.rad, np.arcsin(v3[2])*u.rad]
    
    return c3
    
def test_rotation():
    """"""
    
    c = gc.GD1(phi1=np.linspace(-50,50,100)*u.deg, phi2=np.zeros(100)*u.deg)
    ceq = c.transform_to(coord.ICRS)
    
    c0gd = gc.GD1(phi1=0*u.deg, phi2=90*u.deg)
    c0 = c0gd.transform_to(coord.ICRS)
    c1 = [c0gd.phi1, c0gd.phi2]
    c2 = [c0.ra, c0.dec]
    
    plt.close()
    plt.figure(figsize=(10,5))
    plt.subplot(111) #, projection='mollweide')
    
    #plt.plot(ceq.ra.wrap_at(0*u.deg).rad, ceq.dec.rad, 'r.')
    #print(ceq[0])
    
    for f in np.linspace(0,1,50):
        cpole = great_circle(c1, c2, f=f)
        pole = coord.SkyCoord(ra=cpole[0], dec=cpole[1])
        #print(pole)
        fr = gc.GreatCircleICRSFrame(pole=pole, ra0=0*u.deg)
        
        crot = ceq.transform_to(fr)
        #print(crot[0])
        
        plt.plot(crot.phi1.wrap_at(0*u.deg).rad, crot.phi2.rad, '-', color=mpl.cm.viridis(f))
        
        


def tides(mratio=1/81, dratio=1/60):
    """"""
    #mratio = 1/81
    #dratio = 1/60
    
    aratio = mratio * dratio**2 * (1-2/dratio) / (1-1/dratio)**2
    
    print(aratio)
    
    #print(gp.MilkyWayPotential().mass_enclosed(np.array([-8.3,0,0])*u.kpc))
    #print(gp.MilkyWayPotential().mass_enclosed(np.array([20,0,0])*u.kpc))
