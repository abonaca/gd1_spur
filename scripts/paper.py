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
    
    spur_data = [g['phi1'], g['phi2']]
    
    colors = ['k', 'orange', 'deepskyblue', 'limegreen']
    accent_colors = ['k', 'orangered', 'navy', 'forestgreen']
    sizes = [1, 2, 2, 2]
    
    plt.close()
    fig = plt.figure(figsize=(14,7))
    
    gs0 = mpl.gridspec.GridSpec(1, 2, left=0.07, right=0.97, bottom=0.15, top=0.95, wspace=0.2)

    gs_left = mpl.gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0], width_ratios=[1,2], wspace=0.4, hspace=0.1)
    gs_right = mpl.gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[1], hspace=0.1, wspace=0.1)
    
    # show gap and spur profiles
    for e in range(4):
        ax = plt.Subplot(fig, gs_left[e,0])
        ax1 = fig.add_subplot(ax)
        
        plt.plot(bc, h_data, 'o', ms=6, color=colors[e])
        #plt.errorbar(bc, h_data, yerr=yerr_data, fmt='none', color='k', label='')
        plt.plot(bc, ytop_data, '-', color='k', alpha=0.5, lw=3)
        
        plt.xlim(-55, -25)
        #plt.ylim(-0.5, 1.5)
        if e<3:
            plt.setp(plt.gca().get_xticklabels(), visible=False)
        else:
            plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('Number')
        
        ax = plt.Subplot(fig, gs_left[e,1])
        ax2 = fig.add_subplot(ax)
        
        plt.plot(spur_data[0], spur_data[1], 'o', ms=sizes[e], color=colors[e])
        if e>0:
            ind = (spur_data[0]>-50) & (spur_data[0]<-30)
            plt.plot(spur_data[0][ind], spur_data[1][ind], 'o', ms=2*sizes[e], color=colors[e], mec=accent_colors[e], mew=1)
        
        plt.plot(spx, spy, '-', color='k', alpha=0.5, lw=3)
        
        plt.xlim(-55,-25)
        plt.ylim(-6,6)
        #if e==0:
            #plt.gca().set_aspect('equal')
        
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
    
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar):
            ax_ = plt.Subplot(fig, gs_right[j-1,i])
            ax[j-1][i] = fig.add_subplot(ax_)
            
            vert_id = (pids[:,0]==i) & (pids[:,1]==j)
            xy_vert = vertices[vert_id]
            p = mpl.patches.Polygon(xy_vert, closed=True, lw=2, ec='0.8', fc='0.9', zorder=0)
            plt.gca().add_artist(p)
            
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
    

# calculations

def gd1_width(sig=12*u.arcmin, d=8*u.kpc):
    """Calculate stream width in physical units"""
    
    print((np.arctan(sig.to(u.radian).value)*d).to(u.pc))

def gd1_length():
    """Calculate stream length in physical units"""

def width_range():
    """Find width at different locations along the stream"""
    
    #pickles here: /home/ana/projects/GD1-DR2/notebooks/stream-probs/
