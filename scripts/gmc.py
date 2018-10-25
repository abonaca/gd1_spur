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

def create_streakline(t_impact, seed=3251761, pot='mw'):
    """"""
    np.random.seed(seed)

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
    ind_gap = np.where((model_gd1.phi1.wrap_at(wangle)>-39*u.deg) & (model_gd1.phi1.wrap_at(wangle)<-29*u.deg))[0]

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
    
    return (model, xgap, vgap, ind_gap)

def unperturbed(pot='log'):
    """"""
    np.random.seed(143531)
    
    t_impact = 0.495*u.Gyr
    t_impact = 0.195*u.Gyr
    M = 5e6*u.Msun
    rs = 0.1*rs_diemer(M)
    bnorm = 15*u.pc
    bx = 6*u.pc
    vnorm = 250*u.km/u.s
    vx = -25*u.km/u.s
    
    # potential parameters
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

    # load one orbital point
    pos = np.load('../data/{}_orbit.npy'.format(pot))
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
    
    # load data
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))

    plt.close()
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0)

    plt.sca(ax[0])
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.text(0.03, 0.9, 'Observed GD-1 stream', fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
    txt = plt.text(0.04, 0.75, 'Gaia proper motions\nPanSTARRS photometry', transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
    
    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'k.', ms=3, alpha=0.2)

    plt.xlim(-70, -10)
    plt.ylim(-6,6)
    plt.yticks([-5,0,5])

    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.text(0.03, 0.9, 'Model of a perturbed GD-1', fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
    txt = plt.text(0.04, 0.75, """t = {:.0f} Myr
M = {:.0f}$\cdot$10$^6$ M$_\odot$
$r_s$ = {:.0f} pc
b = {:.0f} pc
V = {:.0f} km s$^{{-1}}$""".format(t_impact.to(u.Myr).value, M.to(u.Msun).value*1e-6, rs.to(u.pc).value, bnorm.to(u.pc).value, vnorm.to(u.km/u.s).value), transform=plt.gca().transAxes, va='top', ha='left', fontsize='x-small', color='0.2')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/stream_encounter_{}.png'.format(pot), dpi=200)

def gmc_perturbation(pot='mw', nimpact=0):
    """"""
    # potential parameters
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
    
    # model at time of impact
    times = disk_crossing_times(pot=pot, verbose=False, graph=False)
    t_impact = times[nimpact]
    model, xgap, vgap, ind_gap = create_streakline(t_impact, pot=pot)
    print(np.linalg.norm(xgap), t_impact)

    vnorm = ham.potential.circular_velocity(xgap)
    bnorm = 10*u.pc
    M = 1e7*u.Msun
    rs = 10*u.pc
    
    #######################
    # Perturber at encounter
    
    i = np.array([1,0,0], dtype=float)
    j = np.array([0,1,0], dtype=float)
    k = np.array([0,0,1], dtype=float)
    
    # find positional plane
    bi = np.cross(j, vgap)
    bi = bi/np.linalg.norm(bi)
    
    bj = np.cross(vgap, bi)
    bj = bj/np.linalg.norm(bj)
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    dt = 0.05*u.Myr
    
    # generate stream model
    potential_perturb = 2
    par_perturb = np.array([M.si.value, rs.si.value, 0, 0, 0])
    
    plt.close()
    Nb = 12
    fig, ax = plt.subplots(Nb, 1, figsize=(12, 2*Nb+0.5), sharex=True, sharey=True)
    
    rasterized = True
    wangle = 180*u.deg
    phi_max = 360*u.deg * (1 - 1/Nb)
    for e, phi in enumerate(np.linspace(0*u.deg, phi_max, Nb)):
        # pick b
        print(phi)
        by = bnorm*np.sin(phi)
        bx = bnorm*np.cos(phi)
        b = bx*bi - by*bj
        xsub = xgap + b
        
        # find circular velocity at this location
        phi = np.arctan2(xsub[1], xsub[0])
        vsub = vnorm*np.sin(phi)*i - vnorm*np.cos(phi)*j
        
        x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
        stream = {}
        stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
        stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
        
        c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
        cg = c.transform_to(gc.GD1)
        
        plt.sca(ax[e])
        plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'k.', ms=3, alpha=0.2)
        
        plt.gca().set_aspect('equal')
        plt.ylabel('$\phi_2$ [deg]')
    
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.xlim(-85,5)
    plt.ylim(-7,5)
    plt.tight_layout()
    plt.savefig('../plots/gd1_gmc_{}_{}.png'.format(pot, nimpact))

def size_mass():
    """Show sizes and masses of known GMCs"""
    
    t = Table.read('../data/gmc.txt', format='cds')
    
    rmin = 12
    outer = t['Rgal']>rmin
    
    plt.close()
    plt.figure(figsize=(6,5))
    
    plt.plot(t['Mf'], t['Rf'], 'o', color='0.5', label='All')
    plt.plot(t['Mf'][outer], t['Rf'][outer], 'ko', label='R > {} kpc'.format(rmin))
    
    #plt.plot(t['Mn'], t['Rn'], 'o', color='0.5', label='All')
    #plt.plot(t['Mn'][outer], t['Rn'][outer], 'ko', label='R > {} kpc'.format(rmin))
    
    plt.legend(frameon=False, loc=2, fontsize='small')
    plt.gca().set_xscale('log')
    plt.xlabel('Mass [$M_\odot$]')
    plt.ylabel('Size [pc]')
    
    
    plt.tight_layout()
    plt.savefig('../plots/gmc_size_mass.png', dpi=200)

