from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gd1 import *
wangle = 180*u.deg

#########
# figures

def param_search():
    """Visualize the process and results of the parameter search"""
    
    # data
    g = Table.read('../data/members.fits')
    
    p = np.load('/home/ana/projects/GD1-DR2/output/polytrack.npy')
    poly = np.poly1d(p)
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

    gap = np.load('../data/gap_properties.npz')
    phi1_edges = gap['phi1_edges']
    gap_position = gap['position']
    gap_width = gap['width']
    gap_yerr = gap['yerr']
    
    ytop_data = tophat(bc, 1, 0, gap_position, gap_width)
    ytop_data = tophat(bc, data_base, data_hat, gap_position, gap_width)

    # spur spline
    sp = np.load('../data/spur_track.npz')
    spx = sp['x']
    spy = sp['y']
    
    # lists with data points
    gap_data = [[bc, h_data]]
    gap_model = [[bc, ytop_data]]
    spur_data = [[g['phi1'], g['phi2']]]
    loops = []
    pfid = []
    
    ids = [-1, 15, 48]
    for i in ids:
        pkl = pickle.load(open('../data/predictions/model_{:03d}.pkl'.format(i), 'rb'))
        cg = pkl['stream']

        gap_data += [[pkl['bincen'], pkl['nbin']]]
        gap_model += [[pkl['bincen'], pkl['nexp']]]
        spur_data += [[cg.phi1.wrap_at(wangle), cg.phi2]]
        loops += [pkl['all_loop']]
        
        # parameters
        x = pkl['x']
        bnorm = np.sqrt(x[1]**2 + x[2]**2)
        vnorm = np.sqrt(x[3]**2 + x[4]**2)
        pfid += [[x[0], bnorm, vnorm, x[6], np.log10(x[5])]]
    
    colors = ['k', 'orange', 'deepskyblue', 'limegreen']
    accent_colors = ['k', 'orangered', 'navy', 'forestgreen']
    colors = ['k', '#ff6600', '#2ca02c', '#37abc8']
    accent_colors = ['k', '#aa4400', '#165016', '#164450']
    colors = ['k', '#2c89a0', '#37abc8', '#5fbcd3']
    accent_colors = ['k', '#164450', '#216778', '#2c89a0']
    sizes = [1, 4, 4, 4]
    labels = ['Data', 'Model A (fiducial)', 'Model B', 'Model C']
    
    plt.close()
    fig = plt.figure(figsize=(13,6.5))
    
    gs0 = mpl.gridspec.GridSpec(1, 2, left=0.07, right=0.97, bottom=0.15, top=0.95, wspace=0.25)

    gs_left = mpl.gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0], width_ratios=[2,1], wspace=0.4, hspace=0.1)
    gs_right = mpl.gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[1], hspace=0.1, wspace=0.1)
    
    # show gap and spur profiles
    for e in range(4):
        ax = plt.Subplot(fig, gs_left[e,1])
        ax1 = fig.add_subplot(ax)
        
        plt.plot(gap_data[e][0], gap_data[e][1], 'o', ms=6, color=colors[e])
        #plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='')
        plt.plot(gap_model[e][0], gap_model[e][1], '-', color='k', alpha=0.5, lw=3)
        
        plt.xlim(-55, -25)
        #plt.ylim(-0.5, 1.5)
        if e<3:
            plt.setp(plt.gca().get_xticklabels(), visible=False)
        else:
            plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('Number')
        
        ax = plt.Subplot(fig, gs_left[e,0])
        ax2 = fig.add_subplot(ax)
        
        plt.plot(spur_data[e][0], spur_data[e][1], 'o', ms=sizes[e], color=colors[e])
        if e>0:
            ind = loops[e-1]
            #print(ind)
            plt.plot(spur_data[e][0][ind], spur_data[e][1][ind], 'o', ms=1.2*sizes[e], color=accent_colors[e], mec=accent_colors[e], mew=1)
        
        plt.plot(spx, spy, '-', color='k', alpha=0.5, lw=3)
        
        plt.xlim(-55,-25)
        plt.ylim(-4,4)
        txt = plt.text(0.07,0.75, labels[e], transform=plt.gca().transAxes, fontsize='small')
        txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
        
        if e<3:
            plt.setp(plt.gca().get_xticklabels(), visible=False)
        else:
            plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('$\phi_2$ [deg]')
    
    
    #########################
    # right side: corner plot
    
    hull = np.load('../data/hull_points_v500w200.npz')
    vertices = hull['vertices']
    pids = hull['panel']
    
    Nvar = 5
    params = ['T [Gyr]', 'b [pc]', 'V [km s$^{-1}$]', '$r_s$ [pc]', 'log M/M$_\odot$']
    lims = [[0.,2], [1,70], [10,500], [1,40], [5,8.6]]
    lims = [[0.,4.5], [0,145], [0,700], [0,99], [4.5,9]]
    ax = [[[]*4]*4]*4
    
    symbols = ['*', 'o', 's']
    sizes = [15, 8, 7]
    labels = ['Model A (fiducial)', 'Model B', 'Model C']
    
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar):
            ax_ = plt.Subplot(fig, gs_right[j-1,i])
            ax[j-1][i] = fig.add_subplot(ax_)
            
            vert_id = (pids[:,0]==i) & (pids[:,1]==j)
            xy_vert = vertices[vert_id]
            p = mpl.patches.Polygon(xy_vert, closed=True, lw=2, ec='0.8', fc='0.9', zorder=0, label='Consistent with data')
            #plt.gca().add_artist(p)
            patch = plt.gca().add_patch(p)
            
            for e in range(3):
                plt.plot(pfid[e][i], pfid[e][j], symbols[e], ms=sizes[e], mec=accent_colors[e+1], mew=1.5, color=colors[e+1], label=labels[e])

            plt.xlim(lims[i])
            plt.ylim(lims[j])
            
            if j-1<3:
                plt.setp(plt.gca().get_xticklabels(), visible=False)
            else:
                plt.xlabel(params[i])
            
            if i>0:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
            else:
                plt.ylabel(params[j])
    
    plt.sca(ax[3][2])
    # sort legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, frameon=False, loc=2, fontsize='medium', bbox_to_anchor=(-0.2,4.3))
    #plt.legend(frameon=False, loc=2, fontsize='medium', bbox_to_anchor=(-0.2,4.3))
    
    plt.savefig('../paper/param_search.pdf')


def kinematic_predictions():
    """Show velocity offsets in three perturbed GD-1 models"""
    
    labels = ['Model A (fiducial)', 'Model B', 'Model C']

    colors = ['orange', 'deepskyblue', 'limegreen']
    accent_colors = ['orangered', 'navy', 'forestgreen']
    colors = ['#ff6600', '#2ca02c', '#37abc8']
    accent_colors = ['#aa4400', '#165016', '#164450']
    
    colors = ['#2c89a0', '#37abc8', '#5fbcd3']
    accent_colors = ['#164450', '#216778', '#2c89a0']
    
    dvr = []
    dmu1 = []
    dmu2 = []
    ddist = []
    loop = []
    
    ids = [-1, 15, 16, 19, 21]
    ids = [-1, 15, 19]
    for e, i in enumerate(ids):
        pkl = pickle.load(open('../data/predictions/model_{:03d}.pkl'.format(i), 'rb'))
        cg = pkl['stream']

        dvr += [[cg.phi1.wrap_at(wangle), pkl['dvr']]]
        dmu1 += [[cg.phi1.wrap_at(wangle), pkl['dmu1']]]
        dmu2 += [[cg.phi1.wrap_at(wangle), pkl['dmu2']]]
        ddist += [[cg.phi1.wrap_at(wangle), pkl['ddist']]]
        
        loop += [pkl['all_loop']]
        loop[e][::8] = True
    
    kinematics = [dvr, dmu1, dmu2, ddist]
    ylabels = ['$\Delta$ $V_r$\n[km s$^{-1}$]', '$\Delta$ $\mu_{\phi_1}$\n[mas yr$^{-1}$]', '$\Delta$ $\mu_{\phi_2}$\n[mas yr$^{-1}$]']
    ylabels = ['$\Delta$ $V_r$ [km s$^{-1}$]', '$\Delta$ $\mu_{\phi_1}$ [mas yr$^{-1}$]', '$\Delta$ $\mu_{\phi_2}$ [mas yr$^{-1}$]', '$\Delta$ d [pc]']
    nrow = 3
    
    plt.close()
    fig, ax = plt.subplots(nrow,3,figsize=(7.5,7.5), sharex=True, sharey='row')
    
    for i in range(nrow):
        for j in range(3):
            plt.sca(ax[i][j])
            
            #plt.plot(kinematics[i][j][0][loop[j]], kinematics[i][j][1][loop[j]], '-', lw=1, color='0.7')
            plt.plot(kinematics[i][j][0][loop[j]], kinematics[i][j][1][loop[j]], 'o', ms=6, color=accent_colors[j])
            plt.plot(kinematics[i][j][0][loop[j]], kinematics[i][j][1][loop[j]], 'o', ms=3.5, color=colors[j])
            
            if i==0:
                plt.title(labels[j], fontsize='medium')
            
            if i==nrow-1:
                plt.xlabel('$\phi_1$ [deg]')
            
            if j==0:
                plt.ylabel(ylabels[i])
            
            plt.xlim(-55,-25)
            plt.xticks([-50,-40,-30])
    
    plt.tight_layout(h_pad=-0.1, w_pad=0.0, rect=[-0.02,0,1,1])
    plt.savefig('../paper/kinematic_predictions.pdf')


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
    plt.figure(figsize=(7,4.5))
    
    # 3, 0.5
    lw = 0.9
    alpha = 0.8
    t = -t

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
        plt.plot(t, rel_distance.to(u.pc), '-', color=mpl.cm.Reds(0.9), alpha=alpha, label=label, lw=lw)
    
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
        plt.plot(t, rel_distance.to(u.pc), '-', color=mpl.cm.Reds(0.7), alpha=alpha, label=label, lw=lw)
    
    # show globulars
    tgc = Table.read('../data/positions_globular_baumgardt.fits')
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
        plt.plot(t, rel_distance.to(u.pc), '-', color=mpl.cm.Reds(0.5), alpha=alpha, label=label, lw=lw)

    plt.plot(t, np.abs(gap_orbit.xyz[2]).to(u.pc), '-', color=mpl.cm.Reds(0.3), alpha=alpha, label='Disk', lw=lw, zorder=0)
    #plt.plot(t, np.sqrt(gap_orbit.xyz[0]**2 + gap_orbit.xyz[1]**2), 'r-', alpha=0.2)

    txt = plt.text(5, 60, 'Maximum permitted impact parameter', va='bottom', ha='right', fontsize='small')
    txt.set_bbox(dict(facecolor='w', alpha=0.8, ec='none'))
    plt.axhline(57, ls='-', color='k', alpha=0.8, lw=1.5, zorder=10)
    
    plt.ylim(30,200000)
    plt.gca().set_yscale('log')
    plt.gca().invert_xaxis()
    
    plt.legend(loc=2, fontsize='small', markerscale=2, handlelength=1)
    plt.xlabel('Lookback time [Myr]')
    plt.ylabel('Relative distance [pc]')
    
    plt.tight_layout()
    plt.savefig('../plots/satellite_distances.png', dpi=200)
    plt.savefig('../paper/satellite_distances.pdf')


def mass_size(nsigma=3):
    """Compilation of masses and sizes of various objects"""
    
    #mpl.rc('text', usetex=True)
    #mpl.rc('text.latex', preamble='\usepackage{color}')
    
    hull = np.load('../data/hull_points_v500w200.npz')
    vertices = hull['vertices']
    vertices[:,1] = 10**vertices[:,1]
    pids = hull['panel']
    params = ['T [Gyr]', 'b [pc]', 'V [km s$^{-1}$]', '$r_s$ [pc]', 'log M/M$_\odot$']
    j = 4
    i = 3

    t = Table.read('../data/gmc.txt', format='cds')
    rmin = 10
    outer = t['Rgal']>rmin
    
    mrange = 10**np.linspace(4, 9, 20)*u.Msun
    rsrange = rs_hernquist(mrange)
    rsrange2 = rs_diemer(mrange)
    
    scatter = 0.16
    rs_low = 10**(-nsigma*scatter) * rsrange2
    rs_high = 10**(nsigma*scatter) * rsrange2
    
    Mhost = 1e12 * u.Msun
    rmin = 13*u.kpc
    rmax = 25*u.kpc
    rs_low = 10**(-nsigma*scatter) * rs_moline(mrange, r=rmin, Mhost=Mhost, verbose=False)
    rs_high = 10**(nsigma*scatter) * rs_moline(mrange, r=rmax, Mhost=Mhost, verbose=False)
    
    print(rs_moline(np.array([1e6])*u.Msun, r=rmin, Mhost=Mhost, verbose=True))
    
    
    #print(10**0.15, 10**-0.15)
    
    ts = Table.read('../data/dwarfs.txt', format='ascii.commented_header')
    
    tgc = Table.read('../data/result_tab.tex', format='latex')
    #tgc.pprint()
    gc_mass = np.array([float(tgc['mass'][x][1:5])*10**float(tgc['mass'][x][-2:-1]) for x in range(len(tgc))])
    
    colors = [mpl.cm.bone(x) for x in [0.15,0.3,0.5,0.7,0.8]]
    accent_colors = [mpl.cm.bone(x-0.2) for x in [0.15,0.3,0.5,0.7,0.8]]
    ms = 6
    accent_ms = 8
    
    plt.close()
    #fig, ax = plt.subplots(1, 2, figsize=(11,6), gridspec_kw={'width_ratios':[1.7,1]})
    fig, ax = plt.subplots(1, 2, figsize=(11,6), gridspec_kw={'width_ratios':[10,1]})
    plt.sca(ax[0])
    
    vert_id = (pids[:,0]==i) & (pids[:,1]==j)
    xy_vert = vertices[vert_id]
    #plt.rc('text', usetex=True)
    #plt.rcParams['text.latex.preamble']=[r'\usepackage{xcolor}']
    p = mpl.patches.Polygon(xy_vert, closed=True, lw=2, ec='0.8', fc='0.9', zorder=0, label='GD-1 perturber\n(Bonaca et al. 2018)')
    patch = plt.gca().add_patch(p)
    
    plt.plot(0.333*t['Rf'][outer], t['Mf'][outer], 'o', color=colors[2], ms=accent_ms, mec=accent_colors[2], mew=1, label='Outer disk molecular clouds\n(Miville-Desch$\^e$nes et al. 2017)')
    plt.plot(0.333*t['Rf'][outer], t['Mf'][outer], 'o', color=colors[2], ms=ms, mec='none', label='')
    
    plt.plot(ts['rh'], ts['mdyn'], 's', color=colors[4], ms=accent_ms, mec=accent_colors[4], mew=1, label='Dwarf galaxies\n(McConnachie 2012)')
    plt.plot(ts['rh'], ts['mdyn'], 's', color=colors[4], ms=ms, mec='none', label='')

    plt.plot(tgc['rhlp'], gc_mass, '^', color=colors[3], ms=accent_ms, mec=accent_colors[3], mew=1, label='Globular clusters\n(Baumgardt & Hilker 2018)')
    plt.plot(tgc['rhlp'], gc_mass, '^', color=colors[3], ms=ms, mec='none', label='')
    
    # lcdm predictions for subhalos
    #plt.fill_betweenx(mrange.value, rs_high.to(u.pc).value, rs_low.to(u.pc).value, color=colors[1], edgecolor='none', linewidth=0, alpha=0.2, label='$\Lambda$CDM subhalos\n(Molin$\\\'e$ et al. 2017)'.format(nsigma))
    plt.fill_betweenx(mrange.value, rs_high.to(u.pc).value, rs_low.to(u.pc).value, color=colors[1], edgecolor='none', linewidth=0, alpha=0.4, label='$\Lambda$CDM subhalos ({:.0f}$\,\sigma$ scatter)\n(Molin$\\\'e$ et al. 2017)'.format(nsigma))
    
    lstyles = [':', ':']
    dashes = [(1,4), (1,8)]
    lw = 1
    alpha = 0.8
    for e, nsigma in enumerate([1, 2]):
        rs_low = 10**(-nsigma*scatter) * rs_moline(mrange, r=rmin, Mhost=Mhost)
        rs_high = 10**(nsigma*scatter) * rs_moline(mrange, r=rmax, Mhost=Mhost)

        plt.plot(rs_low.to(u.pc).value, mrange.value, ls=lstyles[e], color=colors[1], lw=lw, alpha=alpha, label='', dashes=dashes[e], zorder=0)
        plt.plot(rs_high.to(u.pc).value, mrange.value, ls=lstyles[e], color=colors[1], lw=lw, alpha=alpha, label='', dashes=dashes[e], zorder=0)
        #plt.fill_betweenx(mrange.value, rs_high.to(u.pc).value, rs_low.to(u.pc).value, color=colors[1], edgecolor='none', linewidth=0, alpha=0.2, label='')
    
    plt.xlim(1, 1e3)
    plt.ylim(1e4,1e9)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    # customize the order of legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    print(labels)
    order = [3,0,2,1,4]
    handles = [handles[x] for x in order]
    labels = [labels[x] for x in order]
    plt.legend(handles, labels, frameon=False, loc=6, fontsize='medium', bbox_to_anchor=(1.03,0.5), markerscale=1.5, labelspacing=1)
    
    plt.xlabel('Size [pc]')
    plt.ylabel('Mass [M$_\odot$]')
    
    plt.sca(ax[1])
    plt.axis('off')
    
    plt.tight_layout()
    #plt.savefig('../paper/mass_size.pdf')
    #mpl.rc('text', usetex=False)


##############
# calculations

def gd1_width(sig=12*u.arcmin, d=8*u.kpc):
    """Calculate stream width in physical units"""
    
    print((np.arctan(sig.to(u.radian).value)*d).to(u.pc))

def gd1_length():
    """Calculate stream length in physical units"""


def velocity_angles():
    """Check velocity angles for model examples"""
    
    ids = [-1, 15, 19]
    for e, i in enumerate(ids):
        pkl = pickle.load(open('../data/predictions/model_{:03d}.pkl'.format(i), 'rb'))
        p = pkl['params']
        phi = np.arctan2(p[2], p[1])
        #print(phi.to(u.deg))
        phi = np.arctan2(p[4], p[3])
        print(phi.to(u.deg))
        #print(p[3], p[4])

def max_b(p=1):
    """Find maximum impact parameter allowed"""
    
    sampler = np.load('../data/unique_samples_v500w200.npz')
    models = sampler['chain']
    lnp = sampler['lnp']
    pp = np.percentile(lnp, p)
    
    ind = lnp>=pp
    models = models[ind]
    lnp = lnp[ind]
    
    b = np.sqrt(models[:,1]**2 + models[:,2]**2)*u.pc
    
    print('Max b: {:.2f}'.format(np.max(b)))


def einasto():
    """Plot Einasto profile for several values of the shape parameter n"""
    
    r = np.logspace(-2,1.5,100)
    lw = 2
    
    plt.close()
    plt.figure()
    
    for n in [1.7,5]:
        rho = np.exp(-2*n*(r**(1/n)-1))
        plt.plot(r, rho, '-', label='Einasto, n = {}'.format(n), lw=lw)
    
    rho = 4/(r*(1+r)**2)
    plt.plot(r, rho, '-', label='NFW', lw=lw)
    
    rho = 4/(r*(1+r)**3)
    plt.plot(r, rho, '-', label='Hernquist', lw=lw)
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.legend(frameon=False, fontsize='small')
    plt.xlabel('r / r$_{-2}$')
    plt.ylabel('$\\rho$ / $\\rho_{-2}$')
    
    plt.tight_layout()


#########################
# response to the referee

def satellites():
    """"""
    
    tgc = Table.read('../data/result_tab.tex', format='latex')
    #tgc = Table.read('../data/Baumgardt-globclust.fits')
    print(tgc.colnames)
    #print(tgc['Name'])
    
    tv = Table.read('../data/Vasiliev-globclust_space.txt', format='ascii')
    #print(tv.colnames)
    #print(tv['Name'])
    
    tb = Table.read('../data/gc_cat_bersavosh.fits')
    #print(tb.colnames)
    #print(tb['Name'])
    
    vnames = [x.split('(')[0] for x in tv['Name']]
    vaux = [x.split('(')[1][:-1] for x in tv['Name'] if '(' in x]
    
    remain = np.zeros(len(tb), dtype=bool)
    for e, n in enumerate(tb['Name']):
        if n not in vnames and n not in vaux:
            remain[e] = True
    
    cnt = np.sum(remain)
    print(len(tv), len(tb), cnt, len(tb)-cnt)
    print(np.array(tb['Name'][remain]))
    
    rh = np.array([float(x) if len(x)>0 else np.nan for x in tb['rh']])
    mv = np.array([float(x) if len(x)>0 else np.nan for x in tb['M_V,t']])
    rgal = np.array([float(x) if len(x)>0 else np.nan for x in tb['R_galcen']])
    
    missing = ['2MS-GC01', 'GLIMPSE02']
    ind_missing = [True if x in missing else False for x in tb['Name']]
    print(tb['Name'][ind_missing], rh[ind_missing], mv[ind_missing])


def full_gcs():
    """Globular clusters in Gaia, only missing GLIMPSE C02 and 2MASS-GC01 -- both faint/low mass and in the bulge -> unlikely perturbers of GD-1"""
    t = Table.read('../data/baumgardt_position.txt', format='ascii', delimiter=' ')
    t.pprint()
    print(t.colnames)
    
    # store observables
    data = {'name': t['Cluster'], 'ra': t['RA']*u.deg, 'dec': t['DEC']*u.deg, 'distance': t['Rsun']*u.kpc, 'pmra': t['mualpha']*u.mas/u.yr, 'pmdec': t['mu_delta']*u.mas/u.yr, 'vr': t['<RV>']*u.km/u.s}
    
    tout = Table(data=data, names=('name', 'ra', 'dec', 'distance', 'pmra', 'pmdec', 'vr'))
    tout.pprint()
    tout.write('../data/positions_globular_baumgardt.fits', overwrite=True)
    
def audit_satellites():
    """"""
    tcls = Table.read('../data/positions_classical.fits')
    print(len(tcls))
    print(np.sort(np.array(tcls['name'])))
    
    tufd = Table.read('../data/positions_ufd.fits')
    print(len(tufd))
    print(np.sort(np.array(tufd['name'])))

def full_satellites():
    """"""
    t = Table.read('../data/NearbyGalaxies.dat', format='ascii', data_start=31, delimiter=' ')
    t.pprint()

    colnames = ['name', 'rah', 'ram', 'ras', 'decd', 'decm', 'decs', 'ebv', 'dm', 'dm_errm', 'dm_errp', 'vh', 'vh_errm', 'vh_errp', 'vmag', 'vmag_errm', 'vmag_errp', 'pa', 'pa_errm', 'pa_errp', 'e', 'e_errm', 'e_errp', 'mu', 'mu_errm', 'mu_errp', 'rh', 'rh_errm', 'rh_errp', 'sig', 'sig_errm', 'sig_errp', 'vrot', 'vrot_errm', 'vrot_errp', 'mhi', 'sigg', 'sigg_errm', 'sigg_errp', 'vrotg', 'vrotg_errm', 'vrotg_errp', 'feh', 'feh_errm', 'feh_errp', 'flag', 'reference']
    
    for i in range(47):
        t['col{:d}'.format(i+1)].name = colnames[i]
        
    fgc = np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        if t['name'][i][1]=='*':
            fgc[i] = True
            t['name'][i] = t['name'][i][2:-1]
        else:
            t['name'][i] = t['name'][i][1:-1]
    
    t['flag_gc'] = fgc
    
    t.pprint()
    t.write('../data/dwarfs_mcconnachie.fits', overwrite=True)

def noorbit_satellites():
    """Satellites without an orbit determination"""
    
    tall = Table.read('../data/dwarfs_mcconnachie.fits')
    tcls = Table.read('../data/positions_classical.fits')
    tufd = Table.read('../data/positions_ufd.fits')
    
    remain = np.zeros(len(tall), dtype=bool)
    for e, n in enumerate(tall['name']):
        if n not in tcls['name'] and n not in tufd['name']:
            remain[e] = True
    
    print(np.sort(np.array(tall['name'][remain])))
    print(np.sum(remain))
    tr = tall[remain]
    
    Mv = tall['vmag'] - tall['dm']
    d = 10**(0.2*tall['dm']+1)*u.pc
    rh_angle = tall['rh']*u.arcmin
    rh = np.tan(rh_angle.to(u.rad))*d
    
    d0 = 25*u.kpc
    t = 1*u.Gyr
    s = (d - d0)/t
    #print(s[remain].to(u.km/u.s))
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(15,6), sharey=True)
    
    plt.sca(ax[0])
    plt.plot(rh, Mv, 'ko', alpha=0.5)
    plt.plot(rh[remain], Mv[remain], 'ko')
    
    for i in range(len(tr)):
        if rh[remain][i]<1e3*u.pc:
            plt.text(rh[remain][i].value, Mv[remain][i], tall['name'][remain][i], fontsize='small')
    
    plt.axvline(30, ls='-', lw=2, alpha=0.5, color='k')
    
    plt.gca().set_xscale('log')
    plt.xlim(10,1e3)
    plt.gca().invert_yaxis()
    
    plt.sca(ax[1])
    plt.plot(d[remain].to(u.kpc), Mv[remain], 'ko')
    for i in range(len(tr)):
        #if rh[remain][i]<1e3*u.pc:
        plt.text(d[remain][i].to(u.kpc).value, Mv[remain][i], tall['name'][remain][i], fontsize='small')
    
    plt.tight_layout()


# extra figures

def mock_metrics():
    """"""
    pkl = pickle.load(open('../data/fiducial_perturb_python3.pkl', 'rb'))
    cg_ = pkl['cg']
    g = {'phi1': cg_.phi1.wrap_at(180*u.deg).value, 'phi2': cg_.phi2.value}
    
    track = np.load('../data/mock_spur_track.npz')
    x = track['x']
    y = track['y']
    
    xi = np.linspace(-50,-28.5,2000)
    fy = scipy.interpolate.interp1d(x, y, kind='linear')
    #yi = np.interp(xi, x, y)
    yi = fy(xi)
    
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
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,7), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(g['phi1'], g['phi2'], 'k.', ms=3, label='Fiducial model')
    #plt.plot(x, y, 'ro', zorder=0)
    plt.plot(xi, yi, 'k-', alpha=0.3, lw=4, label='Loop track')
    
    plt.legend(markerscale=4)
    plt.xlim(-55,-25)
    plt.ylim(-5,5)
    plt.ylabel('$\phi_2$ [deg]')
    #plt.gca().set_aspect('equal')
    
    plt.sca(ax[1])
    plt.plot(bc, h_data, 'ko', label='Fiducial model')
    plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='')
    plt.plot(bc, ytop_data, '-', color='k', lw=4, alpha=0.3, label='Gap profile')
    
    plt.legend()
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../paper/mock_metrics.png')

def mock_unique(label='_mock_v500w200'):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = trim_chain(sampler['chain'], 200, 0, np.shape(sampler['chain'])[1], nend=5000)
    models, ind = np.unique(chain, axis=0, return_index=True)
    ifinite = np.isfinite(sampler['lnp'][ind])
    
    np.savez('../data/unique_samples5k{}'.format(label), chain=models[ifinite], lnp=sampler['lnp'][ind][ifinite])

def mock_corner(label='_mock_v500w200', p=5, full=False):
    """"""
    sampler = np.load('../data/unique_samples5k{}.npz'.format(label))
    
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
            lims = [[0.,1], [0,145], [0,700], [0,99], [4.5,9]]
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
            
    np.savez('../data/hull_points{}'.format(label), all=hull_ids, unique=np.unique(hull_ids), panel=panel_id, vertices=vertices)
    
    t_impact = 0.495*u.Gyr
    M = 5e6*u.Msun
    rs = 0.1*rs_diemer(M)
    bnorm = 15*u.pc
    vnorm = 250*u.km/u.s
    
    pfid = [t_impact.to(u.Gyr).value, bnorm.to(u.pc).value, vnorm.to(u.km/u.s).value, (rs.to(u.pc).value), np.log10(M.to(u.Msun).value)]
    
    for i in range(Nvar-1):
        for j in range(i+1, Nvar):
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

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../paper/corner{}.png'.format(label))
