from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gd1 import *


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
    wangle = 180*u.deg
    gap_data = [[bc, h_data]]
    gap_model = [[bc, ytop_data]]
    spur_data = [[g['phi1'], g['phi2']]]
    loops = []
    pfid = []
    
    ids = [-1, 15, 16]
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
    sizes = [1, 4, 4, 4]
    labels = ['Data', 'Model A (fiducial)', 'Model B', 'Model C']
    
    plt.close()
    fig = plt.figure(figsize=(13,6.5))
    
    gs0 = mpl.gridspec.GridSpec(1, 2, left=0.07, right=0.97, bottom=0.15, top=0.95, wspace=0.25)

    gs_left = mpl.gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0], width_ratios=[1,2], wspace=0.4, hspace=0.1)
    gs_right = mpl.gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[1], hspace=0.1, wspace=0.1)
    
    # show gap and spur profiles
    for e in range(4):
        ax = plt.Subplot(fig, gs_left[e,0])
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
        
        ax = plt.Subplot(fig, gs_left[e,1])
        ax2 = fig.add_subplot(ax)
        
        plt.plot(spur_data[e][0], spur_data[e][1], 'o', ms=sizes[e], color=colors[e])
        if e>0:
            ind = loops[e-1]
            #print(ind)
            plt.plot(spur_data[e][0][ind], spur_data[e][1][ind], 'o', ms=1.2*sizes[e], color=accent_colors[e], mec=accent_colors[e], mew=1)
        
        plt.plot(spx, spy, '-', color='k', alpha=0.5, lw=3)
        
        plt.xlim(-55,-25)
        plt.ylim(-6,6)
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
    ax = [[[]*4]*4]*4
    
    symbols = ['*', 'o', 's']
    sizes = [15, 8, 8]
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

# calculations

def gd1_width(sig=12*u.arcmin, d=8*u.kpc):
    """Calculate stream width in physical units"""
    
    print((np.arctan(sig.to(u.radian).value)*d).to(u.pc))

def gd1_length():
    """Calculate stream length in physical units"""

def width_range():
    """Find width at different locations along the stream"""
    
    #pickles here: /home/ana/projects/GD1-DR2/notebooks/stream-probs/
