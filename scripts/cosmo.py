from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from colossus.cosmology import cosmology
from colossus.halo import concentration


def cm_relation():
    """"""
    
    cosmology.setCosmology('planck15')
    #for model_name in concentration.models:
        #print(model_name)
    
    #cosmology.setCosmology('bolshoi')
    csm = cosmology.getCurrent()
    M = 10**np.arange(5.0, 12.4, 0.1)
    z_ = 0
    rho_c = csm.rho_c(z_)
    h_ = csm.Hz(z_) * 1e-2
    delta = 200

    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    
    models = ['ludlow16', 'diemer18']
    colors = [mpl.cm.Blues(0.5), mpl.cm.Blues(0.8)]
    #for model_name in concentration.models:
    for e, model_name in enumerate(models):
        c, mask = concentration.concentration(M, '200c', 0.0, model=model_name, range_return=True)
        R = ((3*M)/(4*np.pi*delta*rho_c))**(1/3)
        rs = R / c * 1e3
        
        #plt.plot(M[mask]*h_, rs[mask]*h_, lw=2, label=model_name)
        plt.fill_between(M[mask]*h_, rs[mask]*h_*0.84, rs[mask]*h_*1.16, lw=2, label=model_name, alpha=0.5, color=colors[e])

    #rerkal = 1.05*1e3*np.sqrt(M*h_*1e-8)
    #plt.plot(M*h_, rerkal, 'r-')
    
    plt.axvline(1e6, ls=':', color='0.3')
    plt.axvline(1e7, ls=':', color='0.3')
    plt.axvline(1e8, ls=':', color='0.3')
    
    #plt.xlim(1E3, 4E15)
    #plt.ylim(2.0, 18.0)
    plt.xscale('log')
    plt.yscale('log')
    
    #plt.xlabel('$M_{200,c}$ [$M_\odot$ h$^{-1}$]')
    ##plt.ylabel('Concentration')
    #plt.ylabel('$r_s$ [kpc h$^{-1}$]')
    plt.xlabel('$M_{200,c}$ [$M_\odot$]')
    #plt.ylabel('Concentration')
    plt.ylabel('$r_s$ [pc]')
    plt.legend(fontsize='small', ncol=1, frameon=False, loc=4)
    
    plt.tight_layout()
    plt.savefig('../plots/rs_mass_cdm.png')
