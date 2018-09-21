#include "interact.h"

// Simulation parameters
double Mcli, Mclf, Rcl, dt;

// potential definitions
int par_perpotential[15] = {0, 4, 5, 4, 7, 8, 11, 4, 15, 13, 14, 19, 26, 6, 6};

double energy(double *x, double *v, double vh){
    int i;
    double Ep=0., Ek=0., Etot=0.;
    
    for(i=0;i<3;i++){
        Ek += 0.5*v[i]*v[i];
        Ep += x[i]*x[i];
    }
    Ep = log(Ep) * 0.5 * vh * vh;
    Etot = Ep + Ek;
    
    return Etot;
}

int abinit_interaction(double *xend, double *vend, double dt_, double dt_fine, double T, double Tenc, double Tstream, double Tgap, int Nstream, double *par_pot, int potential, double *par_perturb, int potential_perturb, double bx, double by, double vx, double vy, double *x1, double *x2, double *x3, double *v1, double *v2, double *v3, double *de)
{
    int i, j, k, Nimpact, Nenc, Ntot, Napar_pot, Napar_perturb, Napar_combined, potential_combined;
    double direction, dt_stream, b[3], bi[3], bj[3], binorm, bjnorm, vi[3], vj[3], vinorm, vjnorm, xgap[3], vgap[3], xsub[3], vsub[3], x[3], v[3];
    
    Nimpact = T/dt_;
    dt = dt_;
    
    // setup underlying potential
    Napar_pot = par_perpotential[potential];
    double apar_pot[Napar_pot];
    initpar(potential, par_pot, apar_pot);
    
    // setup perturber potential
    Napar_perturb = par_perpotential[potential_perturb];
    double apar_perturb[Napar_perturb];
    
    // setup combined potential of the galaxy and the perturber
    potential_combined = potential + potential_perturb;
    Napar_combined = par_perpotential[potential_combined];
    double apar_combined[Napar_combined];
    
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    for(i=0;i<Napar_pot;i++)
        apar_combined[i+Napar_perturb] = apar_pot[i];
    
    ///////////////////////////////////////////
    // Find initial perturber & stream position
    direction = -1.;
    
//     double e1, e2;
//     e1 = energy(xgap, vgap, par_pot[0]);
    
    // initial leapfrog step
//     dostep1(xgap, vgap, apar_pot, potential, dt, direction);
    dostep1(xend, vend, apar_pot, potential, dt, direction);

    // leapfrog steps
    for(i=1;i<Nimpact;i++){
//         dostep(xgap, vgap, apar_pot, potential, dt, direction);
        dostep(xend, vend, apar_pot, potential, dt, direction);
    }
    
    // final leapfrog step
//     dostep1(xgap, vgap, apar_pot, potential, dt, -direction);
    dostep1(xend, vend, apar_pot, potential, dt, -direction);
    
//     e2 = energy(xgap, vgap, par_pot[0]);
//     printf("%.20lf %e\n", e1/e2, e1-e2);
    
    /////////////////////////
    // generate stream points
    direction = 1.;
    dt_stream = Tstream / (float)Nstream;
    dt = dt_stream;
    
    // initial leapfrog step
    dostep1(xend, vend, apar_pot, potential, dt, direction);
    t2n(xend, x1, x2, x3, 0);
    t2n(vend, v1, v2, v3, 0);

    // leapfrog steps
    for(i=1;i<Nstream;i++){
        dostep(xend, vend, apar_pot, potential, dt, direction);
        t2n(xend, x1, x2, x3, i);
        t2n(vend, v1, v2, v3, i);
    }
    
    // final leapfrog step
    dostep1(xend, vend, apar_pot, potential, dt, -direction);
    t2n(xend, x1, x2, x3, Nstream-1);
    t2n(vend, v1, v2, v3, Nstream-1);
    
    // find gap location
    int i_;
    i_ = Tgap/dt_stream;
    n2t(xgap, x1, x2, x3, i_);
    n2t(vgap, v1, v2, v3, i_);
    
    // find positional plane
    bi[0] = vgap[2];
    bi[1] = 0.;
    bi[2] = -vgap[0];
    binorm = sqrt(bi[0]*bi[0] + bi[1]*bi[1] + bi[2]*bi[2]);
    
    bj[0] = vgap[1]*bi[2] - vgap[2]*bi[1];
    bj[1] = vgap[2]*bi[0] - vgap[0]*bi[2];
    bj[2] = vgap[0]*bi[1] - vgap[1]*bi[0];
    bjnorm = sqrt(bj[0]*bj[0] + bj[1]*bj[1] + bj[2]*bj[2]);

    // position of the perturber
    for(i=0;i<3;i++){
        bi[i] = bi[i] / binorm;
        bj[i] = bj[i] / bjnorm;
        b[i] = bx*bi[i] + by*bj[i];
        xsub[i] = xgap[i] + b[i];
    }
    
    // find velocity plane
    vi[0] = vgap[1]*b[2] - vgap[2]*b[1];
    vi[1] = vgap[2]*b[0] - vgap[0]*b[2];
    vi[2] = vgap[0]*b[1] - vgap[1]*b[0];
    vinorm = sqrt(vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);
    
    vj[0] = b[1]*vi[2] - b[2]*vi[1];
    vj[1] = b[2]*vi[0] - b[0]*vi[2];
    vj[2] = b[0]*vi[1] - b[1]*vi[0];
    vjnorm = sqrt(vj[0]*vj[0] + vj[1]*vj[1] + vj[2]*vj[2]);
    
    // velocity of the perturber
    for(i=0;i<3;i++){
        vi[i] = vi[i] / vinorm;
        vj[i] = vj[i] / vjnorm;
        vsub[i] = vx*vi[i] + vy*vj[i];
    }
    
    //////////////////////////////////////////////////
    // Find initial positions for perturber and stream
    direction = -1.;
    Nenc = Tenc / dt_fine;
    dt = dt_fine;
    Ntot = (T + Tenc) / dt_fine + 1;
    
    // initial leapfrog step
    // perturber
    dostep1(xsub, vsub, apar_pot, potential, dt, direction);
    // stream
    for(i=0;i<Nstream;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }

    // leapfrog steps
    for(j=1;j<Nenc;j++){
        // perturber
        dostep(xsub, vsub, apar_pot, potential, dt, direction);
        // stream
        for(i=0;i<Nstream;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_pot, potential, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
        }
    }
    
    // final leapfrog step
    // perturber
    dostep1(xsub, vsub, apar_pot, potential, dt, -direction);
    // stream
    for(i=0;i<Nstream;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, -direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
        
        de[i] = energy(x, v, par_pot[0]);
    }
    
    ///////////////////
    // Perturb the tube
    direction = 1.;
    
//     // Reinitiate the perturber
//     t2t(x0, xp);
//     t2t(v0, vp);
    
    ////////////////////////
    // Initial leapfrog step
    for(k=0;k<3;k++)
        par_perturb[k+Napar_perturb-3] = xsub[k];
    initpar(potential_perturb, par_perturb, apar_perturb);
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstream;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_combined, potential_combined, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
    // update perturber
    dostep1(xsub, vsub, apar_pot, potential, dt, direction);
    
    /////////////////
    // Leapfrog steps
    // in the combined potential of the perturber and the Galaxy
    for(j=1;j<2*Nenc+1;j++){
//     for(j=1;j<Ntot;j++){
        for(k=0;k<3;k++)
            par_perturb[k+Napar_perturb-3] = xsub[k];
        initpar(potential_perturb, par_perturb, apar_perturb);
        for(i=0;i<Napar_perturb;i++)
            apar_combined[i] = apar_perturb[i];
        
        // update stream points
        for(i=0;i<Nstream;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_combined, potential_combined, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
        }
        
        // update perturber
        dostep(xsub, vsub, apar_pot, potential, dt, direction);
    }
    
    // update stream points
    for(i=0;i<Nstream;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, -direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
    // without the perturber, to avoid double encounters
    dt = dt_;
    Ntot = (T-Tenc)/dt_;
    
    // update stream points
    for(i=0;i<Nstream;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
    for(j=0;j<Ntot;j++){
        // update stream points
        for(i=0;i<Nstream;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_pot, potential, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
        }
    }
    
    //////////////////////
    // Final leapfrog step
//     for(k=0;k<3;k++)
//         par_perturb[k+Napar_perturb-3] = xsub[k];
//     initpar(potential_perturb, par_perturb, apar_perturb);
//     for(i=0;i<Napar_perturb;i++)
//         apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstream;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, -direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
        
        de[i] -= energy(x, v, par_pot[0]);
    }
    
    return 0;
}

int general_interact(double *par_perturb, double *x0, double *v0, double Tenc, double T, double dt_, double *par_pot, int potential, int potential_perturb, int Nstar, double *x1, double *x2, double *x3, double *v1, double *v2, double *v3)
{
    int i, j, k, Nenc, Ntot, Napar_perturb, Napar_pot, potential_combined, Napar_combined;
    double x[3], v[3], xp[3], vp[3], direction;
    FILE *fp = fopen("interaction.bin", "wb");
    
    // setup underlying potential
    Napar_pot = par_perpotential[potential];
    double apar_pot[Napar_pot];
    initpar(potential, par_pot, apar_pot);
    
    // setup perturber potential
    Napar_perturb = par_perpotential[potential_perturb];
    double apar_perturb[Napar_perturb];
    
    // setup combined potential of the galaxy and the perturber
    potential_combined = potential + potential_perturb;
    Napar_combined = par_perpotential[potential_combined];
    double apar_combined[Napar_combined];
    
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    for(i=0;i<Napar_pot;i++)
        apar_combined[i+Napar_perturb] = apar_pot[i];
    
    Nenc = Tenc / dt_;
    Ntot = (T + Tenc) / dt_ + 1;
    dt = dt_;
    
//     // perturber at encounter
//     x0[0] = 0.;
//     x0[1] = B*cos(phi);
//     x0[2] = B*sin(phi);
//     
//     v0[0] = V*cos(theta);
//     v0[1] = 0.;
//     v0[2] = V*sin(theta);

//     printf("%e %e %e\n", x0[0], x0[1], x0[2]);
//     printf("%e %e %e\n", v0[0], v0[1], v0[2]);

    //////////////////////////////////
    // Find initial perturber position
//     apar[0] = 0.;
    direction = -1.;
    
//     for(i=0;i<3;i++){
//         printf("%e %e\n", x0[i], v0[i]);
//     }
    
    // initial leapfrog step
    dostep1(x0, v0, apar_pot, potential, dt, direction);

    // leapfrog steps
    for(i=1;i<Nenc;i++){
        dostep(x0, v0, apar_pot, potential, dt, direction);
//         printf("%d %e %e %e\n", i, x0[0], x0[1], x0[2]);
//         printf("%d %e %e %e\n", i, v0[0], v0[1], v0[2]);
    }
    
    // final leapfrog step
    dostep1(x0, v0, apar_pot, potential, dt, direction);
    
//     printf("%e %e %e\n", x0[0], x0[1], x0[2]);
//     printf("%e %e %e\n", v0[0], v0[1], v0[2]);
    
    ///////////////////////////////
    // Find initial stream position
    direction = -1.;
    
//     // Reinitiate the perturber
//     t2t(x0, xp);
//     t2t(v0, vp);
    
    ////////////////////////
    // Initial leapfrog step
//     for(k=0;k<3;k++)
//         par_perturb[k+Napar_perturb-3] = xp[k];
//     initpar(potential_perturb, par_perturb, apar_perturb);
//     for(i=0;i<Napar_perturb;i++)
//         apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
//     printf("0 %e %e\n", x[0], x[1]);

    
//     // update perturber
//     dostep1(xp, vp, apar_pot, potential, dt, direction);
    
    /////////////////
    // Leapfrog steps
    for(j=1;j<Nenc;j++){
//         for(k=0;k<3;k++)
//             par_perturb[k+Napar_perturb-3] = xp[k];
//         initpar(potential_perturb, par_perturb, apar_perturb);
//         for(i=0;i<Napar_perturb;i++)
//             apar_combined[i] = apar_perturb[i];
        
        // update stream points
        for(i=0;i<Nstar;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_pot, potential, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
            
//             if (j>Nenc && j%1000==0){
//                 fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x1[i], x2[i], x3[i], v1[i], v2[i], v3[i]);
//                 fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x[0], x[1], x[2], v[0], v[1], v[2]);
//             }
        }
//         printf("%d %e %e\n", j, x[0], x[1]);

//         // update perturber
//         dostep(xp, vp, apar_pot, potential, dt, direction);
    }
    
    //////////////////////
    // Final leapfrog step
//     for(k=0;k<3;k++)
//         par_perturb[k+Napar_perturb-3] = xp[k];
//     initpar(potential_perturb, par_perturb, apar_perturb);
//     for(i=0;i<Napar_perturb;i++)
//         apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
    ///////////////////
    // Perturb the tube
    direction = 1.;
    
    // Reinitiate the perturber
    t2t(x0, xp);
    t2t(v0, vp);
    
    ////////////////////////
    // Initial leapfrog step
    for(k=0;k<3;k++)
        par_perturb[k+Napar_perturb-3] = xp[k];
    initpar(potential_perturb, par_perturb, apar_perturb);
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
//     for(i=0;i<3;i++){
//         printf("%e %e\n", xp[i], vp[i]);
//     }
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_combined, potential_combined, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
//     printf("0 %e %e\n", x[0], x[1]);

    
    // update perturber
    dostep1(xp, vp, apar_pot, potential, dt, direction);
    
    /////////////////
    // Leapfrog steps
    for(j=1;j<Ntot;j++){
        for(k=0;k<3;k++)
            par_perturb[k+Napar_perturb-3] = xp[k];
        initpar(potential_perturb, par_perturb, apar_perturb);
        for(i=0;i<Napar_perturb;i++)
            apar_combined[i] = apar_perturb[i];
        
        // update stream points
        for(i=0;i<Nstar;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_combined, potential_combined, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
            
            if (j>Nenc && j%1000==0){
                fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x1[i], x2[i], x3[i], v1[i], v2[i], v3[i]);
//                 fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x[0], x[1], x[2], v[0], v[1], v[2]);
            }
        }
//         printf("%d %e %e\n", j, x[0], x[1]);

        // update perturber
        dostep(xp, vp, apar_pot, potential, dt, direction);
    }
    
    //////////////////////
    // Final leapfrog step
    for(k=0;k<3;k++)
        par_perturb[k+Napar_perturb-3] = xp[k];
    initpar(potential_perturb, par_perturb, apar_perturb);
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_combined, potential_combined, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
//     printf("%e %e %e\n", xp[0], xp[1], xp[2]);
//     printf("%e %e %e\n", vp[0], vp[1], vp[2]);
    
//     // update perturber
//     dostep1(xp, vp, apar_pot, potential, dt, direction);
    
    fclose(fp);
    
    return 0;
}

int interact(double *par_perturb, double B, double phi, double V, double theta, double Tenc, double T, double dt_, double *par_pot, int potential, int potential_perturb, int Nstar, double *x1, double *x2, double *x3, double *v1, double *v2, double *v3)
{
    int i, j, k, Nenc, Ntot, Napar_perturb, Napar_pot, potential_combined, Napar_combined;
    double x[3], v[3], x0[3], v0[3], xp[3], vp[3], direction;
    FILE *fp = fopen("interaction.bin", "wb");
    
    // setup underlying potential
    Napar_pot = par_perpotential[potential];
    double apar_pot[Napar_pot];
    initpar(potential, par_pot, apar_pot);
    
    // setup perturber potential
    Napar_perturb = par_perpotential[potential_perturb];
    double apar_perturb[Napar_perturb];
    
    // setup combined potential of the galaxy and the perturber
    potential_combined = potential + potential_perturb;
    Napar_combined = par_perpotential[potential_combined];
    double apar_combined[Napar_combined];
    
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    for(i=0;i<Napar_pot;i++)
        apar_combined[i+Napar_perturb] = apar_pot[i];
    
    Nenc = Tenc / dt_;
    Ntot = (T + Tenc) / dt_ + 1;
    dt = dt_;
    
    // perturber at encounter
    x0[0] = 0.;
    x0[1] = B*cos(phi);
    x0[2] = B*sin(phi);
    
    v0[0] = V*cos(theta);
    v0[1] = 0.;
    v0[2] = V*sin(theta);

//     printf("%e %e %e\n", x0[0], x0[1], x0[2]);
//     printf("%e %e %e\n", v0[0], v0[1], v0[2]);

    //////////////////////////////////
    // Find initial perturber position
//     apar[0] = 0.;
    direction = -1.;
    
    // initial leapfrog step
    dostep1(x0, v0, apar_pot, potential, dt, direction);

    // leapfrog steps
    for(i=1;i<Nenc;i++){
        dostep(x0, v0, apar_pot, potential, dt, direction);
//         printf("%d %e %e %e\n", i, x0[0], x0[1], x0[2]);
//         printf("%d %e %e %e\n", i, v0[0], v0[1], v0[2]);
    }
    
    // final leapfrog step
    dostep1(x0, v0, apar_pot, potential, dt, direction);
    
//     printf("%e %e %e\n", x0[0], x0[1], x0[2]);
//     printf("%e %e %e\n", v0[0], v0[1], v0[2]);
    
    ///////////////////////////////
    // Find initial stream position
    direction = -1.;
    
//     // Reinitiate the perturber
//     t2t(x0, xp);
//     t2t(v0, vp);
    
    ////////////////////////
    // Initial leapfrog step
//     for(k=0;k<3;k++)
//         par_perturb[k+Napar_perturb-3] = xp[k];
//     initpar(potential_perturb, par_perturb, apar_perturb);
//     for(i=0;i<Napar_perturb;i++)
//         apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
//     printf("0 %e %e\n", x[0], x[1]);

    
//     // update perturber
//     dostep1(xp, vp, apar_pot, potential, dt, direction);
    
    /////////////////
    // Leapfrog steps
    for(j=1;j<Nenc;j++){
//         for(k=0;k<3;k++)
//             par_perturb[k+Napar_perturb-3] = xp[k];
//         initpar(potential_perturb, par_perturb, apar_perturb);
//         for(i=0;i<Napar_perturb;i++)
//             apar_combined[i] = apar_perturb[i];
        
        // update stream points
        for(i=0;i<Nstar;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_pot, potential, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
            
//             if (j>Nenc && j%1000==0){
//                 fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x1[i], x2[i], x3[i], v1[i], v2[i], v3[i]);
//                 fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x[0], x[1], x[2], v[0], v[1], v[2]);
//             }
        }
//         printf("%d %e %e\n", j, x[0], x[1]);

//         // update perturber
//         dostep(xp, vp, apar_pot, potential, dt, direction);
    }
    
    //////////////////////
    // Final leapfrog step
//     for(k=0;k<3;k++)
//         par_perturb[k+Napar_perturb-3] = xp[k];
//     initpar(potential_perturb, par_perturb, apar_perturb);
//     for(i=0;i<Napar_perturb;i++)
//         apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_pot, potential, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
    ///////////////////
    // Perturb the tube
    direction = 1.;
    
    // Reinitiate the perturber
    t2t(x0, xp);
    t2t(v0, vp);
    
    ////////////////////////
    // Initial leapfrog step
    for(k=0;k<3;k++)
        par_perturb[k+Napar_perturb-3] = xp[k];
    initpar(potential_perturb, par_perturb, apar_perturb);
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_combined, potential_combined, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
//     printf("0 %e %e\n", x[0], x[1]);

    
    // update perturber
    dostep1(xp, vp, apar_pot, potential, dt, direction);
    
    /////////////////
    // Leapfrog steps
    for(j=1;j<Ntot;j++){
        for(k=0;k<3;k++)
            par_perturb[k+Napar_perturb-3] = xp[k];
        initpar(potential_perturb, par_perturb, apar_perturb);
        for(i=0;i<Napar_perturb;i++)
            apar_combined[i] = apar_perturb[i];
        
        // update stream points
        for(i=0;i<Nstar;i++){
            // choose a star
            n2t(x, x1, x2, x3, i);
            n2t(v, v1, v2, v3, i);
            
            dostep(x, v, apar_combined, potential_combined, dt, direction);
            
            t2n(x, x1, x2, x3, i);
            t2n(v, v1, v2, v3, i);
            
            if (j>Nenc && j%1000==0){
                fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x1[i], x2[i], x3[i], v1[i], v2[i], v3[i]);
//                 fprintf(fp, "%d %d %f %f %f %f %f %f\n", j, i, x[0], x[1], x[2], v[0], v[1], v[2]);
            }
        }
//         printf("%d %e %e\n", j, x[0], x[1]);

        // update perturber
        dostep(xp, vp, apar_pot, potential, dt, direction);
    }
    
    //////////////////////
    // Final leapfrog step
    for(k=0;k<3;k++)
        par_perturb[k+Napar_perturb-3] = xp[k];
    initpar(potential_perturb, par_perturb, apar_perturb);
    for(i=0;i<Napar_perturb;i++)
        apar_combined[i] = apar_perturb[i];
    
    // update stream points
    for(i=0;i<Nstar;i++){
        // choose a star
        n2t(x, x1, x2, x3, i);
        n2t(v, v1, v2, v3, i);
        
        dostep1(x, v, apar_combined, potential_combined, dt, direction);
        
        t2n(x, x1, x2, x3, i);
        t2n(v, v1, v2, v3, i);
    }
    
//     printf("%e %e %e\n", xp[0], xp[1], xp[2]);
//     printf("%e %e %e\n", vp[0], vp[1], vp[2]);
    
//     // update perturber
//     dostep1(xp, vp, apar_pot, potential, dt, direction);
    
    fclose(fp);
    
    return 0;
}

int encounter(double M, double B, double phi, double V, double theta, double T, double dt_, double *x1, double *x2, double *x3, double *v1, double *v2, double *v3)
{
    int i, N, Ntot, potential;
    double x[3], v[3], direction, apar[1];
    
    N = T /dt_;
    Ntot = 2*N + 1;
    printf("%e %d %d\n", T/dt_, N, Ntot);
    
    // perturber at encounter
    x[0] = 0.;
    x[1] = B*cos(phi);
    x[2] = B*sin(phi);
    
    v[0] = V*cos(theta);
    v[1] = 0.;
    v[2] = V*sin(theta);

    ///////////////////////////////////////
    // Orbit integration
    
    // Free space
    potential = 0;
    apar[0] = 0.;
    
    // Time step
    dt = dt_;
    
    ///////////
    // Backward
    direction = -1.;
    t2n(x, x1, x2, x3, N);
    t2n(v, v1, v2, v3, N);

    // initial leapfrog step
    dostep1(x, v, apar, potential, dt, direction);
    
    // Record
    t2n(x, x1, x2, x3, N-1);
    t2n(v, v1, v2, v3, N-1);

    // leapfrog steps
    for(i=1;i<N;i++){
        dostep(x, v, apar, potential, dt, direction);

        // Record
        t2n(x, x1, x2, x3, N-1-i);
        t2n(v, v1, v2, v3, N-1-i);
    }
    
    // final leapfrog step
    dostep1(x,v,apar,potential,dt,direction);
    
    // Record
    t2n(x, x1, x2, x3, 0);
    t2n(v, v1, v2, v3, 0);
    
    //////////
    // Forward
    direction = 1.;
    n2t(x, x1, x2, x3, N);
    n2t(v, v1, v2, v3, N);

    // initial leapfrog step
    dostep1(x, v, apar, potential, dt, direction);
    
    // Record
    t2n(x, x1, x2, x3, N+1);
    t2n(v, v1, v2, v3, N+1);

    // leapfrog steps
    for(i=1;i<N;i++){
        dostep(x, v, apar, potential, dt, direction);

        // Record
        t2n(x, x1, x2, x3, N+1+i);
        t2n(v, v1, v2, v3, N+1+i);
    }
    
    // final leapfrog step
    dostep1(x,v,apar,potential,dt,direction);
    
    // Record
    t2n(x, x1, x2, x3, 2*N);
    t2n(v, v1, v2, v3, 2*N);

    for(i=0;i<3;i++){
        printf("%d %e %e\n", i, x[i], v[i]);
    }
    
    return 0;
}

int stream(double *x0, double *v0, double *xm1, double *xm2, double *xm3, double *xp1, double *xp2, double *xp3, double *vm1, double *vm2, double *vm3, double *vp1, double *vp2, double *vp3, double *par, double *offset, int potential, int integrator, int N, int M, double mcli, double mclf, double rcl, double dt_)
{
	int i,j, k=0, Napar, Ne, imin=0;
	double x[3], v[3], xs[3], vs[3], omega[3], om, sign=1., back=-1., r, rp, rm, vtot, vlead, vtrail, dM, Mcl, dR, dRRj, time=0.;
	double *xc1, *xc2, *xc3, *Rj, *dvl, *dvt;
	long s1=560;
    
    double xlmc[3], vlmc[3] = {262000, 465000, 56000};
    if (potential==6){
        for(j=0;j<3;j++){
            xlmc[j] = par[12+j];
//             printf("%e ", xlmc[j]);
        }
//         printf("\n");
    }
	
	// number of output particles
	Ne=ceil((float)N/(float)M);
	xc1 = (double*) malloc(N*sizeof(double));
	xc2 = (double*) malloc(N*sizeof(double));
	xc3 = (double*) malloc(N*sizeof(double));
	Rj = (double*) malloc(Ne*sizeof(double));
	dvl = (double*) malloc(Ne*sizeof(double));
	dvt = (double*) malloc(Ne*sizeof(double));
	
	// Initial position and velocities
	t2t(x0,x);
	t2t(v0,v);
	
	// Initial mass and mass loss rate
	Mcli = mcli;
	Mclf = mclf;
	Mcl = Mcli;
	dM = (Mcli-Mclf)/(double)N;
	
	// Cluster size
	Rcl = rcl;
	
	// Position offset
	dR = offset[0];
	
	// Initialize velocity offsets, drawn from a Maxwell distribution
	double r1,r2,r3;
	for(i=0;i<Ne;i++){
		// Leading tail velocity offsets
		r1=gasdev(&s1);
		r2=gasdev(&s1);
		r3=gasdev(&s1);
		dvl[i]=sqrt(r1*r1 + r2*r2 + r3*r3)*offset[1]/3.;
		
		// Trailing tail velocity offsets
		r1=gasdev(&s1);
		r2=gasdev(&s1);
		r3=gasdev(&s1);
		dvt[i]=sqrt(r1*r1 + r2*r2 + r3*r3)*offset[1]/3.;
	}
	
	// Set up actual potential parameters;
	Napar = par_perpotential[potential];
	double apar[Napar], apar_aux[11];
	initpar(potential, par, apar);
    
    if(potential==6){
        for(i=0;i<11;i++)
            apar_aux[i] = apar[i];
    }
	
	// Integrator switch
	void (*pt2dostep)(double*, double*, double*, int, double, double) = NULL;
	
	if(integrator==0){
		// Leapfrog
		pt2dostep=&dostep;
	}
	else if(integrator==1){
		// Runge-Kutta
		pt2dostep=&dostep_rk;
	}
	
	// Time step
	dt = dt_;
	
	///////////////////////////////////////
	// Backward integration (cluster only)
    
	if(integrator==0){
		dostep1(x,v,apar,potential,dt,back);
		imin=1;
        time = time + dt*back;
        
        if(potential==6){
            dostep1(xlmc,vlmc,apar_aux,4,dt,back);
            for(j=0;j<3;j++){
                apar[12+j] = xlmc[j];
//                 printf("%e ", xlmc[i]);
            }
//             printf("\n");
        }
	}
	for(i=imin;i<N;i++){
		(*pt2dostep)(x,v,apar,potential,dt,back);
        time = time + dt*back;
        
        if(potential==6){
            (*pt2dostep)(xlmc,vlmc,apar_aux,4,dt,back);
            for(j=0;j<3;j++){
                apar[12+j] = xlmc[j];
//                 printf("%e ", xlmc[j]);
            }
//             printf("%d\n", i);
        }
	}
	if(integrator==0){
		dostep1(x,v,apar,potential,dt,back);
        if(potential==6){
            dostep1(xlmc,vlmc,apar_aux,4,dt,back);
            for(i=0;i<3;i++){
                apar[12+i] = xlmc[i];
//                 printf("%e ", xlmc[i]);
            }
//             printf("\n");
        }
    }
    
//     printf("%e", time);
	
	////////////////////////////////////////////
	// Forward integration (cluster and stream)
	
	// Initial step for the leapfrog integrator
	if (integrator==0){
		dostep1(x,v,apar,potential,dt,sign);
		for(j=0;j<3;j++)
			x[j]=x[j]-dt*v[j];
        
        if(potential==6){
            dostep1(xlmc,vlmc,apar_aux,4,dt,sign);
            for(j=0;j<3;j++)
                xlmc[j]=xlmc[j]-dt*vlmc[j];
            for(j=0;j<3;j++)
                apar[12+j] = xlmc[j];
        }
	
		dostep(x,v,apar,potential,dt,sign);
		imin=1;
        
        if(potential==6){
            dostep(xlmc,vlmc,apar_aux,4,dt,sign);
            for(i=0;j<3;j++)
                apar[12+j] = xlmc[j];
        }
		
		// Update output arrays
		t2n(x, xc1, xc2, xc3, 0);
		Rj[k]=jacobi(x, v, apar, potential, Mcl);	// Jacobi radius
		r=len(x);
		rm=(r-Rj[k])/r;
		rp=(r+Rj[k])/r;
		
		// Angular velocity
		omega[0]=x[1]*v[2]-x[2]*v[1];
		omega[1]=x[2]*v[0]-x[0]*v[2];
		omega[2]=x[0]*v[1]-x[1]*v[0];
		om=len(omega)/(r*r);
		vtot=len(v);
		vlead=(vtot-om*Rj[0])/vtot;
		vtrail=(vtot+om*Rj[0])/vtot;
		
		dvl[k]/=r;
		dvt[k]/=r;
		
		// Inner particle
		xm1[k]=x[0]*rm + dR*Rj[k]*gasdev(&s1);
		xm2[k]=x[1]*rm + dR*Rj[k]*gasdev(&s1);
		xm3[k]=x[2]*rm + dR*Rj[k]*gasdev(&s1);
		vm1[k]=v[0]*vlead - dvl[k]*x[0];
		vm2[k]=v[1]*vlead - dvl[k]*x[1];
		vm3[k]=v[2]*vlead - dvl[k]*x[2];
		
		// Outer particle
		xp1[k]=x[0]*rp + dR*Rj[k]*gasdev(&s1);
		xp2[k]=x[1]*rp + dR*Rj[k]*gasdev(&s1);
		xp3[k]=x[2]*rp + dR*Rj[k]*gasdev(&s1);
		vp1[k]=v[0]*vtrail + dvt[k]*x[0];
		vp2[k]=v[1]*vtrail + dvt[k]*x[1];
		vp3[k]=v[2]*vtrail + dvt[k]*x[2];
		k++;
        
        time = time + dt*sign;
	}

	// Subsequent steps
	for(i=imin;i<N;i++){
		Mcl-=dM;
		
		(*pt2dostep)(x,v,apar,potential,dt,sign);
        
        if(potential==6){
            (*pt2dostep)(xlmc,vlmc,apar_aux,4,dt,sign);
            for(j=0;j<3;j++)
                apar[12+j] = xlmc[j];
        }

        // Store cluster position
		t2n(x, xc1, xc2, xc3, i);
		
		// Propagate previously released stream particles
		for(j=0;j<k;j++){
			// Inner particle
			n2t(xs, xm1, xm2, xm3, j);
			n2t(vs, vm1, vm2, vm3, j);
			dostep_stream(x,xs,vs,apar,potential,Mcl,dt,sign);
// 			(*pt2dostep)(xs,vs,apar,potential,dt,sign);
			
			// Update
			t2n(xs, xm1, xm2, xm3, j);
			t2n(vs, vm1, vm2, vm3, j);
			
			// Outer particle
			n2t(xs, xp1, xp2, xp3, j);
			n2t(vs, vp1, vp2, vp3, j);
			dostep_stream(x,xs,vs,apar,potential,Mcl,dt,sign);
// 			(*pt2dostep)(xs,vs,apar,potential,dt,sign);
			
			// Update
			t2n(xs, xp1, xp2, xp3, j);
			t2n(vs, vp1, vp2, vp3, j);
		}
		
		if(i%M==0){
			// Release only at every Mth timestep
			// Jacobi tidal radius
			Rj[k]=jacobi(x, v, apar, potential, Mcl);
			r=len(x);
			rm=(r-Rj[k])/r;
			rp=(r+Rj[k])/r;
			
			// Angular velocity
			omega[0]=x[1]*v[2]-x[2]*v[1];
			omega[1]=x[2]*v[0]-x[0]*v[2];
			omega[2]=x[0]*v[1]-x[1]*v[0];
			om=len(omega)/(r*r);
			vtot=len(v);
			vlead=(vtot-om*Rj[k])/vtot;
			vtrail=(vtot+om*Rj[k])/vtot;
			
			dvl[k]/=r;
			dvt[k]/=r;
			
			// Generate 2 new stream particles at the tidal radius
			dRRj = dR*Rj[k];
			// Inner particle (leading tail)
			xm1[k]=x[0]*rm + dRRj*gasdev(&s1);
			xm2[k]=x[1]*rm + dRRj*gasdev(&s1);
			xm3[k]=x[2]*rm + dRRj*gasdev(&s1);
			vm1[k]=v[0]*vlead - dvl[k]*x[0];
			vm2[k]=v[1]*vlead - dvl[k]*x[1];
			vm3[k]=v[2]*vlead - dvl[k]*x[2];
			
			// Outer particle (trailing tail)
			xp1[k]=x[0]*rp + dRRj*gasdev(&s1);
			xp2[k]=x[1]*rp + dRRj*gasdev(&s1);
			xp3[k]=x[2]*rp + dRRj*gasdev(&s1);
			vp1[k]=v[0]*vtrail + dvt[k]*x[0];
			vp2[k]=v[1]*vtrail + dvt[k]*x[1];
			vp3[k]=v[2]*vtrail + dvt[k]*x[2];
			k++;
		}
		
		time = time + dt*sign;
	}
	
//     printf("%e\n", time);
	
    if (integrator==0){
		dostep1(x,v,apar,potential,dt,back);
    
        if(potential==6){
            dostep1(xlmc,vlmc,apar_aux,4,dt,back);
            for(i=0;i<3;i++){
                apar[12+i] = xlmc[i];
//                 printf("%e ", xlmc[i]);
            }
//             printf("\n");
        }
    }
	
	// Free memory
	free(xc1);
	free(xc2);
	free(xc3);
	free(Rj);
	free(dvl);
	free(dvt);
	
	return 0;
} 

int orbit(double *x0, double *v0, double *x1, double *x2, double *x3, double *v1, double *v2, double *v3, double *par, int potential, int integrator, int N, double dt_, double direction)
{
	int i, Napar, imin=0;
	double x[3], v[3];
	
	// Initial position and velocities
	t2t(x0, x);
	t2t(v0, v);
	
    // Set up actual potential parameters;
    Napar = par_perpotential[potential];
	double apar[Napar];
	initpar(potential, par, apar);
	
	// Integrator switch
	void (*pt2dostep)(double*, double*, double*, int, double, double) = NULL;
	
	if(integrator==0){
		// Leapfrog
		pt2dostep=&dostep;
	}
	else if(integrator==1){
		// Runge-Kutta
		pt2dostep=&dostep_rk;
	}
	
	// Time step
	dt = dt_;
	
	///////////////////////////////////////
	// Orbit integration
	
	if(integrator==0){
		dostep1(x,v,apar,potential,dt,direction);
		
		// Record
		t2n(x, x1, x2, x3, 0);
		t2n(v, v1, v2, v3, 0);
		imin=1;
	}
	for(i=imin;i<N;i++){
		(*pt2dostep)(x,v,apar,potential,dt,direction);
		
		// Record
		t2n(x, x1, x2, x3, i);
		t2n(v, v1, v2, v3, i);
	}
	if(integrator==0){
		dostep1(x,v,apar,potential,dt,direction);
		
		// Record
		t2n(x, x1, x2, x3, N-1);
		t2n(v, v1, v2, v3, N-1);
	}
	
	return 0;
}

void dostep(double *x, double *v, double *par, int potential, double deltat, double sign)
{	// Evolve point particle from x0, v0, for a time deltat in a given potential
	// evolve forward for sign=1, backwards for sign=-1
	// return final positions and velocities in x, v
	
	int i, j, Nstep;
	double xt[3], vt[3], at[3], dts;
	
	dts=sign*dt;			// Time step with a sign
	Nstep=(int) (deltat/dt);	// Number of steps to evolve

	for(i=0;i<Nstep;i++){
		// Forward the particle using the leapfrog integrator
		for(j=0;j<3;j++)
			xt[j]=x[j]+dts*v[j];
		force(xt, at, par, potential);

        for(j=0;j<3;j++){
			vt[j]=v[j]+dts*at[j];
//             if(potential==1)
//                 printf(" %e %e ", at[j], vt[j]);
        }
//         if(potential==1)
//             printf("\n");
		
		// Update input vectors to current values
		for(j=0;j<3;j++){
			x[j]=xt[j];
			v[j]=vt[j];
		}
	}
	
}

void dostep1(double *x, double *v, double *par, int potential, double deltat, double sign)
{	// Make first step to set up the leapfrog integration
	
	double a[3], dts;
	
	dts=sign*dt;
	force(x, a, par, potential);

	v[0]=v[0]+0.5*dts*a[0];
	v[1]=v[1]+0.5*dts*a[1];
	v[2]=v[2]+0.5*dts*a[2];
}

void dostep_rk(double *x, double *v, double *par, int potential, double deltat, double sign)
{	// Evolve point particle from x0, v0, for a time deltat in a given potential
	// evolve forward for sign=1, backwards for sign=-1
	// return final positions and velocities in x, v
	// prototype for Runge-Kutta integrator
	
	int i;
	double xt1[3], xt2[3], xt3[3], vt1[3], vt2[3], vt3[3], a[3], at1[3], at2[3], at3[3], dts, dt2;
	
	dts=sign*dt;			// Time step with a sign
	dt2=dts/2.;
	
	// Initial values
	force(x, a, par, potential);
	
	// First half-step
	for(i=0;i<3;i++){
		xt1[i]=x[i]+dt2*v[i];
		vt1[i]=v[i]+dt2*a[i];
	}
	force(xt1,at1,par,potential);
	
	// Second half-step
	for(i=0;i<3;i++){
		xt2[i]=x[i]+dt2*vt1[i];
		vt2[i]=v[i]+dt2*at1[i];
	}
	force(xt2,at2,par,potential);
	
	// Third step
	for(i=0;i<3;i++){
		xt3[i]=x[i]+dts*vt2[i];
		vt3[i]=v[i]+dts*at2[i];
	}
	force(xt3,at3,par,potential);
	
	// Final Runge-Kutta evaluation
	for(i=0;i<3;i++){
		x[i]+=dts/6.*(v[i]+2.*(vt1[i]+vt2[i])+vt3[i]);
		v[i]+=dts/6.*(a[i]+2.*(at1[i]+at2[i])+at3[i]);
	}
	
}

void dostep_stream(double *xc, double *x, double *v, double *par, int potential, double Mcl, double deltat, double sign)
{	// Same as dostep, except that stream particles also feel the Plummer potential from a cluster
	
	int i, j, Nstep;
	double xt[3], vt[3], at[3], xr[3], ar[3], dts;
	
	dts=sign*dt;			// Time step with a sign
	Nstep=(int) (deltat/dt);	// Number of steps to evolve

	for(i=0;i<Nstep;i++){
		// Forward the particle using the leapfrog integrator
		for(j=0;j<3;j++){
			xt[j]=x[j]+dts*v[j];
			xr[j]=xc[j]-xt[j];
		}
		force(xt, at, par, potential);
//         if(potential==7) par[12]+=dts;
        force_plummer(xr,ar,Mcl);
		for(j=0;j<3;j++)
			vt[j]=v[j]+dts*(at[j]+ar[j]);
		
		// Update input vectors to current values
		for(j=0;j<3;j++){
			x[j]=xt[j];
			v[j]=vt[j];
		}
	}
}

void force(double *x, double *a, double *par, int potential)
{
	int i;
	double r, aux, aux2;
	
    if(potential==0){
        for(i=0;i<3;i++)
            a[i] = 0.;
    }
	else if(potential==1){
		// Point mass potential
		// par = [Mtot, x, y, z]
		r=sqrt((x[0]-par[1])*(x[0]-par[1]) + (x[1]-par[2])*(x[1]-par[2]) + (x[2]-par[3])*(x[2]-par[3]));
		aux=-par[0]/(r*r*r);
        
		for(i=0;i<3;i++)
			a[i]=aux*(x[i]-par[i+1]);

//         printf("%e\n", a[0]);

	}else if(potential==2){
        // Hernquist spheroid
        // par = [Mtot, a, x, y, z]
		r=sqrt((x[0]-par[2])*(x[0]-par[2]) + (x[1]-par[3])*(x[1]-par[3]) + (x[2]-par[4])*(x[2]-par[4]));
		aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
		
        for(i=0;i<3;i++)
            a[i]=aux*(x[i]-par[i+2]);
        
    }else if(potential==3){
		// Logarithmic potential
		// par = [Vc^2, q, q^2, rhalo^2]
		r=x[0]*x[0] + x[1]*x[1] + x[2]*x[2]/(par[2])+par[3];
		aux=-par[0]/r;
		
		a[0]=aux*x[0];
		a[1]=aux*x[1];
		a[2]=aux*x[2]/(par[1]);
//         printf("%e %e\n", par[0], a[0]);
		
	}else if(potential==4){
        // Combination of a logarithmic and point mass potential
        double a1[3], a2[3], par1[4], par2[4];
        int p1, p2;
        p1 = 1;
        p2 = 3;
        
        for(i=0;i<4;i++)
            par1[i] = par[i];
        
        for(i=0;i<4;i++)
            par2[i] = par[i+4];
        
        force(x, a1, par1, p1);
        force(x, a2, par2, p2);
        
        for(i=0;i<3;i++)
            a[i] = a1[i] + a2[i];
//         printf("pert %e %e %e\n", a1[0], a1[1], a1[2]);
//         printf("gal %e %e %e\n", a2[0], a2[1], a2[2]);
        
    }else if(potential==5){
        // Combination of a logarithmic and hernquist potential
        double a1[3], a2[3], par1[5], par2[3];
        int p1, p2;
        p1 = 2;
        p2 = 3;
        
        for(i=0;i<5;i++)
            par1[i] = par[i];
        
        for(i=0;i<4;i++)
            par2[i] = par[i+5];
        
        force(x, a1, par1, p1);
        force(x, a2, par2, p2);
        
        for(i=0;i<3;i++)
            a[i] = a1[i] + a2[i];
    }else if(potential==41){
		// Triaxial logarithmic halo potential from Law & Majewski (2010)
		// par = [Vc^2, c1, c2, c3, c4, rhalo^2]
		r=par[1]*x[0]*x[0] + par[2]*x[1]*x[1] + par[3]*x[0]*x[1] + par[4]*x[2]*x[2] + par[5];
		aux=-par[0]/r;
		
		a[0]=aux*(2*par[1]*x[0] + par[3]*x[1]);
		a[1]=aux*(2*par[2]*x[1] + par[3]*x[0]);
		a[2]=aux*(2*par[4]*x[2]);
		
	}else if(potential==51){
		// Triaxial NFW halo potential, parameters similar to Law & Majewski (2010)
		// par = [GM, c1, c2, c3, c4, rhalo]
		r=sqrt(par[1]*x[0]*x[0] + par[2]*x[1]*x[1] + par[3]*x[0]*x[1] + par[4]*x[2]*x[2]);
		aux=0.5 * par[0] / (r*r*r) * (1./(1.+par[5]/r)-log(1.+r/par[5]));
		
		a[0]=aux*(2*par[1]*x[0] + par[3]*x[1]);
		a[1]=aux*(2*par[2]*x[1] + par[3]*x[0]);
		a[2]=aux*(2*par[4]*x[2]);
		
	}else if(potential==6){
		// Composite Galactic potential featuring a disk, bulge, and flattened NFW halo (from Johnston/Law/Majewski/Helmi)
		// par = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo]
		
		//Hernquist bulge
		r=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
		aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
		
		a[0]=aux*x[0];
		a[1]=aux*x[1];
		a[2]=aux*x[2];
		
		//Miyamoto-Nagai disk
		aux2=sqrt(x[2]*x[2] + par[4]);
		r=sqrt(x[0]*x[0] + x[1]*x[1] + (par[3] + aux2) * (par[3] + aux2));
		aux=-par[2]/(r*r*r);
		
		a[0]+=aux*x[0];
		a[1]+=aux*x[1];
		a[2]+=aux*x[2]*(par[3] + aux2)/aux2;
		
		//Triaxial NFW Halo
		r=sqrt(par[6]*x[0]*x[0] + par[7]*x[1]*x[1] + par[8]*x[0]*x[1] + par[9]*x[2]*x[2]);
		aux=0.5 * par[5]/(r*r*r) * (1./(1.+par[10]/r)-log(1.+r/par[10]));
		
		a[0]+=aux*(2*par[6]*x[0] + par[8]*x[1]);
		a[1]+=aux*(2*par[7]*x[1] + par[8]*x[0]);
		a[2]+=aux*(2*par[9]*x[2]);
		
	}else if(potential==7){
		// Spherical NFW potential
		// par = [GM, Rh]
		r=sqrt(x[0]*x[0] + x[1]*x[1]*par[2] + x[2]*x[2]*par[3]);
		aux=par[0]/(r*r*r) * (1./(1.+par[1]/r)-log(1.+r/par[1]));
		
		a[0]=aux*x[0];
		a[1]=aux*x[1]*par[2];
		a[2]=aux*x[2]*par[3];

    }else if(potential==8){
        // Galactic potential + LMC
        // par = [GMb, ab, GMd, ad, bd^2, GM, q^2, rhalo, GMlmc, Xlmc, Ylmc, Zlmc]
        
        //Hernquist bulge
        r=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
        
        a[0]=aux*x[0];
        a[1]=aux*x[1];
        a[2]=aux*x[2];
        
        //Miyamoto disk
        aux2=sqrt(x[2]*x[2] + par[4]);
        r=sqrt(x[0]*x[0] + x[1]*x[1] + (par[3] + aux2) * (par[3] + aux2));
        aux=-par[2]/(r*r*r);
        
        a[0]+=aux*x[0];
        a[1]+=aux*x[1];
        a[2]+=aux*x[2]*(par[3] + aux2)/aux2;
        
        //Triaxial NFW Halo
        r=sqrt(par[6]*x[0]*x[0] + par[7]*x[1]*x[1] + par[8]*x[0]*x[1] + par[9]*x[2]*x[2]);
        aux=0.5 * par[5]/(r*r*r) * (1./(1.+par[10]/r)-log(1.+r/par[10]));
        
        a[0]+=aux*(2*par[6]*x[0] + par[8]*x[1]);
        a[1]+=aux*(2*par[7]*x[1] + par[8]*x[0]);
        a[2]+=aux*(2*par[9]*x[2]);
        
        // Point mass
        // added softening ~8pc, assuming X_LMC~-0.8kpc
        r=sqrt((x[0]-par[12])*(x[0]-par[12]) + (x[1]-par[13])*(x[1]-par[13]) + (x[2]-par[14])*(x[2]-par[14])) - 0.01*par[12];
        aux = par[11]/(r*r*r);
        
        a[0]+=aux*x[0];
        a[1]+=aux*x[1];
        a[2]+=aux*x[2];
        
    }else if(potential==9){
        // Galactic potential + LMC on a string
        // par = [GMb, ab, GMd, ad, bd^2, GM, q^2, rhalo, Mlmc, t]
        
        //Hernquist bulge
        r=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
        
        a[0]=aux*x[0];
        a[1]=aux*x[1];
        a[2]=aux*x[2];
        
        //Miyamoto disk
        aux2=sqrt(x[2]*x[2] + par[4]);
        r=sqrt(x[0]*x[0] + x[1]*x[1] + (par[3] + aux2) * (par[3] + aux2));
        aux=-par[2]/(r*r*r);
        
        a[0]+=aux*x[0];
        a[1]+=aux*x[1];
        a[2]+=aux*x[2]*(par[3] + aux2)/aux2;
        
        //Triaxial NFW Halo
        r=sqrt(par[6]*x[0]*x[0] + par[7]*x[1]*x[1] + par[8]*x[0]*x[1] + par[9]*x[2]*x[2]);
        aux=0.5 * par[5]/(r*r*r) * (1./(1.+par[10]/r)-log(1.+r/par[10]));
        
        a[0]+=aux*(2*par[6]*x[0] + par[8]*x[1]);
        a[1]+=aux*(2*par[7]*x[1] + par[8]*x[0]);
        a[2]+=aux*(2*par[9]*x[2]);
        
        //LMC on a string
        double xlmc[3]={-2.509654716638902e19, -1.2653311505262738e21, -8.319850498177284e20};
        double vlmc[3]={-57, -226, 221};
        r=sqrt((x[0]-xlmc[0]-vlmc[0]*par[12])*(x[0]-xlmc[0]-vlmc[0]*par[12]) + (x[1]-xlmc[1]-vlmc[1]*par[12])*(x[1]-xlmc[1]-vlmc[1]*par[12]) + (x[2]-xlmc[2]-vlmc[2]*par[12])*(x[2]-xlmc[2]-vlmc[2]*par[12]));
        aux=-par[11]/(r*r*r);
        
        a[0]+=aux*x[0];
        a[1]+=aux*x[1];
        a[2]+=aux*x[2];
    }else if(potential==10){
		// Composite Galactic potential featuring a disk, bulge, flattened NFW halo (from Johnston/Law/Majewski/Helmi) and perturbations from dipole expansion
		// par = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, a10, a11, a12]
		
		//Hernquist bulge
		r=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
		aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
		
		a[0]=aux*x[0];
		a[1]=aux*x[1];
		a[2]=aux*x[2];
		
		//Miyamoto-Nagai disk
		aux2=sqrt(x[2]*x[2] + par[4]);
		r=sqrt(x[0]*x[0] + x[1]*x[1] + (par[3] + aux2) * (par[3] + aux2));
		aux=-par[2]/(r*r*r);
		
		a[0]+=aux*x[0];
		a[1]+=aux*x[1];
		a[2]+=aux*x[2]*(par[3] + aux2)/aux2;
		
		//Triaxial NFW Halo
		r=sqrt(par[6]*x[0]*x[0] + par[7]*x[1]*x[1] + par[8]*x[0]*x[1] + par[9]*x[2]*x[2]);
		aux=0.5 * par[5]/(r*r*r) * (1./(1.+par[10]/r)-log(1.+r/par[10]));
		
		a[0]+=aux*(2*par[6]*x[0] + par[8]*x[1]);
		a[1]+=aux*(2*par[7]*x[1] + par[8]*x[0]);
		a[2]+=aux*(2*par[9]*x[2]);
        
        // Dipole moment
        a[0]+=par[13];
        a[1]+=par[11];
        a[2]+=par[12];
    }else if(potential==11){
		// Composite Galactic potential featuring a disk, bulge, flattened NFW halo (from Johnston/Law/Majewski/Helmi) and perturbations from dipole and quadrupole moment
		// par = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, a10, a11, a12, a20, a21, a22, a23, a24]
		
		//Hernquist bulge
		r=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
		aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
		
		a[0]=aux*x[0];
		a[1]=aux*x[1];
		a[2]=aux*x[2];
		
		//Miyamoto-Nagai disk
		aux2=sqrt(x[2]*x[2] + par[4]);
		r=sqrt(x[0]*x[0] + x[1]*x[1] + (par[3] + aux2) * (par[3] + aux2));
		aux=-par[2]/(r*r*r);
		
		a[0]+=aux*x[0];
		a[1]+=aux*x[1];
		a[2]+=aux*x[2]*(par[3] + aux2)/aux2;
		
		//Triaxial NFW Halo
		r=sqrt(par[6]*x[0]*x[0] + par[7]*x[1]*x[1] + par[8]*x[0]*x[1] + par[9]*x[2]*x[2]);
		aux=0.5 * par[5]/(r*r*r) * (1./(1.+par[10]/r)-log(1.+r/par[10]));
		
		a[0]+=aux*(2*par[6]*x[0] + par[8]*x[1]);
		a[1]+=aux*(2*par[7]*x[1] + par[8]*x[0]);
		a[2]+=aux*(2*par[9]*x[2]);
        
        // Dipole moment
        a[0]+=par[13];
        a[1]+=par[11];
        a[2]+=par[12];
        
        // Quadrupole moment
        a[0]+= x[0]*(par[18] - par[16]) + x[1]*par[14] + x[2]*par[17];
        a[1]+= x[0]*par[14] - x[1]*(par[16] + par[18]) + x[2]*par[15];
        a[2]+= x[0]*par[17] + x[1]*par[15] + x[2]*2*par[16];
    }else if(potential==12){
		// Composite Galactic potential featuring a disk, bulge, flattened NFW halo (from Johnston/Law/Majewski/Helmi) and perturbations from dipole, quadrupole and octupole moments
		// par = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, a10, a11, a12, a20, a21, a22, a23, a24, a30, a31, a32, a33, a34, a35, a36]
		
		//Hernquist bulge
		r=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
		aux=-par[0]/(r * (r+par[1]) * (r+par[1]));
		
		a[0]=aux*x[0];
		a[1]=aux*x[1];
		a[2]=aux*x[2];
		
		//Miyamoto-Nagai disk
		aux2=sqrt(x[2]*x[2] + par[4]);
		r=sqrt(x[0]*x[0] + x[1]*x[1] + (par[3] + aux2) * (par[3] + aux2));
		aux=-par[2]/(r*r*r);
		
		a[0]+=aux*x[0];
		a[1]+=aux*x[1];
		a[2]+=aux*x[2]*(par[3] + aux2)/aux2;
		
		//Triaxial NFW Halo
		r=sqrt(par[6]*x[0]*x[0] + par[7]*x[1]*x[1] + par[8]*x[0]*x[1] + par[9]*x[2]*x[2]);
		aux=0.5 * par[5]/(r*r*r) * (1./(1.+par[10]/r)-log(1.+r/par[10]));
		
		a[0]+=aux*(2*par[6]*x[0] + par[8]*x[1]);
		a[1]+=aux*(2*par[7]*x[1] + par[8]*x[0]);
		a[2]+=aux*(2*par[9]*x[2]);
        
        // Dipole moment
        a[0]+=par[13];
        a[1]+=par[11];
        a[2]+=par[12];
        
        // Quadrupole moment
        a[0]+= x[0]*(par[18] - par[16]) + x[1]*par[14] + x[2]*par[17];
        a[1]+= x[0]*par[14] - x[1]*(par[16] + par[18]) + x[2]*par[15];
        a[2]+= x[0]*par[17] + x[1]*par[15] + x[2]*2*par[16];
        
        // Octupole moment
        a[0]+= par[19]*6.*x[0]*x[1] + par[20]*x[1]*x[2] + par[21]*(-2.*x[0]*x[1]) + par[22]*(-6.*x[0]*x[2]) + par[23]*(4.*x[2]*x[2] - x[1]*x[1] - 3.*x[0]*x[0]) + par[24]*2.*x[0]*x[2] + par[25]*3.*(x[0]*x[0] - x[1]*x[1]);
        a[1]+= par[19]*3.*(x[0]*x[0] - x[1]*x[1]) + par[20]*x[0]*x[2] + par[21]*(4.*x[2]*x[2] - x[0]*x[0] - 3.*x[1]*x[1]) + par[22]*(-6.*x[1]*x[2]) + par[23]*(-2.*x[0]*x[1]) + par[24]*(-2.*x[1]*x[2]) + par[25]*(-6.*x[0]*x[1]);
        a[2]+= par[20]*x[0]*x[1] + par[21]*8.*x[1]*x[2] + par[22]*(6.*x[2]*x[2] - 3.*x[0]*x[0] - 3.*x[1]*x[1]) + par[23]*8*x[0]*x[2] + par[24]*(x[0]*x[0] - x[1]*x[1]);
    }
}

void force_plummer(double *x, double *a, double Mcl)
{	// Calculate acceleration a at a position x from a cluster with a Plummer profile
	// Assumes global definitions of cluster mass Mcl and radius Rcl
	int i;
	double r, raux;
	
	r=len(x);
	raux=pow(r*r+Rcl*Rcl, 1.5);
	
	for(i=0;i<3;i++)
		a[i]=G*Mcl*x[i]/raux;
}

void initpar(int potential, double *par, double *apar)
{
	if(potential==1){
		// Point mass potential, par = [Mtot, x, y, z]
		// apar = [G*Mtot, x, y, z]
		apar[0]=G*par[0];
		apar[1]=par[1];
		apar[2]=par[2];
		apar[3]=par[3];
		
    }else if(potential==2){
        // Hernquist spheroidal, par = [Mtot, a, x, y, z]
        // apar = [G*Mtot, a, x, y, z]
        apar[0]=G*par[0];
        apar[1]=par[1];
        apar[2]=par[2];
        apar[3]=par[3];
        apar[4]=par[4];
        
    }else if(potential==3){
		// Logarithmic potential, par = [Vc, q_phi, rhalo]
		// apar = [Vc^2, q, q^2, rhalo^2]
		apar[0]=par[0]*par[0];
		apar[1]=par[1];
		apar[2]=par[1]*par[1];
        apar[3]=par[2]*par[2];
		
	}else if(potential==4){
		// Triaxial halo potential from Law & Majewski (2010)
		// par = [Vc, phi, q_1, q_2, q_z, rhalo]
		// apar = [Vc^2, c1, c2, c3, c4, rhalo^2]
		double cosphi, sinphi;
		
		cosphi=cos(par[1]);
		sinphi=sin(par[1]);
		
		apar[0]=par[0]*par[0];
		apar[1]=cosphi*cosphi/(par[2]*par[2]) + sinphi*sinphi/(par[3]*par[3]);
		apar[2]=cosphi*cosphi/(par[3]*par[3]) + sinphi*sinphi/(par[2]*par[2]);
		apar[3]=2*sinphi*cosphi*(1/(par[2]*par[2]) - 1/(par[3]*par[3]));
		apar[4]=1/(par[4]*par[4]);
		apar[5]=par[5]*par[5];
		
	}else if(potential==5){
		// Triaxial NFW halo potential from Law & Majewski (2010)
		// par = [V, rhalo, phi, q_1, q_2, q_z]
		// apar = [GM, c1, c2, c3, c4, rhalo]
		double cosphi, sinphi;
		
		cosphi=cos(par[2]);
		sinphi=sin(par[2]);
		
// 		apar[0]=G*Msun*pow(10,par[0]);
		apar[0]=par[0]*par[0]*par[1];
		apar[1]=cosphi*cosphi/(par[3]*par[3]) + sinphi*sinphi/(par[4]*par[4]);
		apar[2]=cosphi*cosphi/(par[4]*par[4]) + sinphi*sinphi/(par[3]*par[3]);
		apar[3]=2*sinphi*cosphi*(1/(par[3]*par[3]) - 1/(par[4]*par[4]));
		apar[4]=1/(par[5]*par[5]);
		apar[5]=par[1];

	}else if(potential==6){
		// Composite Galactic potential featuring a disk, bulge, and triaxial NFW halo (from Johnston/Law/Majewski/Helmi)
		// par = [GMb, ab, GMd, ad, bd, V, rhalo, phi, q_1, q_2, q_z]
		// apar = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo]
		double cosphi, sinphi; //, tq, tphi;
		
		apar[0]=G*par[0];
		apar[1]=par[1];
		apar[2]=G*par[2];
		apar[3]=par[3];
		apar[4]=par[4]*par[4];
		
		cosphi=cos(par[7]);
		sinphi=sin(par[7]);
		
		apar[5]=par[5]*par[5]*par[6];
		apar[6]=cosphi*cosphi/(par[8]*par[8]) + sinphi*sinphi/(par[9]*par[9]);
		apar[7]=cosphi*cosphi/(par[9]*par[9]) + sinphi*sinphi/(par[8]*par[8]);
		apar[8]=2*sinphi*cosphi*(1/(par[8]*par[8]) - 1/(par[9]*par[9]));
		apar[9]=1/(par[10]*par[10]);
		apar[10]=par[6];

    }else if(potential==7){
		apar[0]=G*Msun*pow(10,par[0]);
		apar[1]=par[1];
		apar[2]=par[5]*par[5];
		apar[3]=par[6]*par[6];

	}else if(potential==8){
        // Galactic potential + LMC
        // par = [GMb, ab, GMd, ad, bd, Vh, rhalo, phi, q1, q2, qz, Mlmc, Xlmc, Ylmc, Zlmc]
        // apar = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, GMlmc, Xlmc, Ylmc, Zlmc]
        double cosphi, sinphi;

        apar[0]=G*par[0];
        apar[1]=par[1];
        
        apar[2]=G*par[2];
        apar[3]=par[3];
        apar[4]=par[4]*par[4];
        
        cosphi=cos(par[7]);
        sinphi=sin(par[7]);
        apar[5]=par[5]*par[5]*par[6];
        apar[6]=cosphi*cosphi/(par[8]*par[8]) + sinphi*sinphi/(par[9]*par[9]);
        apar[7]=cosphi*cosphi/(par[9]*par[9]) + sinphi*sinphi/(par[8]*par[8]);
        apar[8]=2*sinphi*cosphi*(1/(par[8]*par[8]) - 1/(par[9]*par[9]));
        apar[9]=1/(par[10]*par[10]);
        apar[10]=par[6];
        
        apar[11]=G*par[11];
        apar[12]=par[12];
        apar[13]=par[13];
        apar[14]=par[14];
    }else if(potential==9){
        // Galactic potential + LMC on a string
        // par = [GMb, ab, GMd, ad, bd, V, rhalo, phi, q_1, q_2, q_z, Mlmc]
        // apar = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, Mlmc, t]
        double cosphi, sinphi; //, tq, tphi;
        
        apar[0]=G*par[0];
        apar[1]=par[1];
        apar[2]=G*par[2];
        apar[3]=par[3];
        apar[4]=par[4]*par[4];
        
        cosphi=cos(par[7]);
        sinphi=sin(par[7]);
        apar[5]=par[5]*par[5]*par[6];
        apar[6]=cosphi*cosphi/(par[8]*par[8]) + sinphi*sinphi/(par[9]*par[9]);
        apar[7]=cosphi*cosphi/(par[9]*par[9]) + sinphi*sinphi/(par[8]*par[8]);
        apar[8]=2*sinphi*cosphi*(1/(par[8]*par[8]) - 1/(par[9]*par[9]));
        apar[9]=1/(par[10]*par[10]);
        apar[10]=par[6];
        apar[11]=G*par[11];
//         printf("%e\n", par[11]);
        apar[12]=0.;
    }else if(potential==10){
		// Composite Galactic potential featuring a disk, bulge, and triaxial NFW halo (from Johnston/Law/Majewski/Helmi)
		// par = [GMb, ab, GMd, ad, bd, V, rhalo, phi, q_1, q_2, q_z, a10, a11, a12]
		// apar = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, fa10, fa11, fa12]
		double cosphi, sinphi, f; //, tq, tphi;
		
		apar[0]=G*par[0];
		apar[1]=par[1];
		apar[2]=G*par[2];
		apar[3]=par[3];
		apar[4]=par[4]*par[4];
		
		cosphi=cos(par[7]);
		sinphi=sin(par[7]);
		
		apar[5]=par[5]*par[5]*par[6];
		apar[6]=cosphi*cosphi/(par[8]*par[8]) + sinphi*sinphi/(par[9]*par[9]);
		apar[7]=cosphi*cosphi/(par[9]*par[9]) + sinphi*sinphi/(par[8]*par[8]);
		apar[8]=2*sinphi*cosphi*(1/(par[8]*par[8]) - 1/(par[9]*par[9]));
		apar[9]=1/(par[10]*par[10]);
		apar[10]=par[6];
        
        f = sqrt(3./(4*pi));
        apar[11] = f*par[11];
        apar[12] = f*par[12];
        apar[13] = f*par[13];
    }else if(potential==11){
		// Composite Galactic potential featuring a disk, bulge, and triaxial NFW halo (from Johnston/Law/Majewski/Helmi)
		// par = [GMb, ab, GMd, ad, bd, V, rhalo, phi, q_1, q_2, q_z, a10, a11, a12, a20, a21, a22, a23, a24]
		// apar = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, fa10, fa11, fa12, fa20, fa21, fa22, fa23, fa24]
		double cosphi, sinphi, f; //, tq, tphi;
		
		apar[0]=G*par[0];
		apar[1]=par[1];
		apar[2]=G*par[2];
		apar[3]=par[3];
		apar[4]=par[4]*par[4];
		
		cosphi=cos(par[7]);
		sinphi=sin(par[7]);
		
		apar[5]=par[5]*par[5]*par[6];
		apar[6]=cosphi*cosphi/(par[8]*par[8]) + sinphi*sinphi/(par[9]*par[9]);
		apar[7]=cosphi*cosphi/(par[9]*par[9]) + sinphi*sinphi/(par[8]*par[8]);
		apar[8]=2*sinphi*cosphi*(1/(par[8]*par[8]) - 1/(par[9]*par[9]));
		apar[9]=1/(par[10]*par[10]);
		apar[10]=par[6];
        
        f = sqrt(3./(4*pi));
        apar[11] = f*par[11];
        apar[12] = f*par[12];
        apar[13] = f*par[13];
        
        f = 0.5*sqrt(15./pi);
        apar[14] = f*par[14];
        apar[15] = f*par[15];
        apar[16] = f/sqrt(3.)*par[16];
        apar[17] = f*par[17];
        apar[18] = f*par[18];
    }else if(potential==12){
        // Composite Galactic potential featuring a disk, bulge, and triaxial NFW halo (from Johnston/Law/Majewski/Helmi)
		// par = [GMb, ab, GMd, ad, bd, V, rhalo, phi, q_1, q_2, q_z, a10, a11, a12, a20, a21, a22, a23, a24, a30, a31, a32, a33, a34, a35, a36]
		// apar = [GMb, ab, GMd, ad, bd^2, GM, c1, c2, c3, c4, rhalo, fa10, fa11, fa12, fa20, fa21, fa22, fa23, fa24, fa30, fa31, fa32, fa33, fa34, fa35, fa36]
		double cosphi, sinphi, f; //, tq, tphi;
		
		apar[0]=G*par[0];
		apar[1]=par[1];
		apar[2]=G*par[2];
		apar[3]=par[3];
		apar[4]=par[4]*par[4];
		
		cosphi=cos(par[7]);
		sinphi=sin(par[7]);
		
		apar[5]=par[5]*par[5]*par[6];
		apar[6]=cosphi*cosphi/(par[8]*par[8]) + sinphi*sinphi/(par[9]*par[9]);
		apar[7]=cosphi*cosphi/(par[9]*par[9]) + sinphi*sinphi/(par[8]*par[8]);
		apar[8]=2*sinphi*cosphi*(1/(par[8]*par[8]) - 1/(par[9]*par[9]));
		apar[9]=1/(par[10]*par[10]);
		apar[10]=par[6];
        
        // dipole
        f = sqrt(3./(4*pi));
        apar[11] = f*par[11];
        apar[12] = f*par[12];
        apar[13] = f*par[13];
        
        // quadrupole
        f = 0.5*sqrt(15./pi);
        apar[14] = f*par[14];
        apar[15] = f*par[15];
        apar[16] = f/sqrt(3.)*par[16];
        apar[17] = f*par[17];
        apar[18] = f*par[18];
        
        // octupole
        f = 0.25*sqrt(35./(2.*pi));
        apar[19] = f*par[19];
        apar[25] = f*par[25];
        
        f = 0.25*sqrt(105./pi);
        apar[20] = f*par[20]*2.;
        apar[24] = f*par[24];
        
        f = 0.25*sqrt(21./(2.*pi));
        apar[21] = f*par[21];
        apar[23] = f*par[23];
        
        apar[22] = 0.25*sqrt(7./pi)*par[22];
    }
}

double jacobi(double *x, double *v, double *par, int potential, double Mcl)
{	// Jacobi radius of a cluster
	// at the position x, velocity v, and in a given potential
	int i;
	double R, om, dpot, delta, r;
	double omega[3], x1[3], x2[3], a1[3], a2[3], dx[3]; //, da[3];

	// Radial distance
	r=len(x);
	
	// Angular velocity
	omega[0]=x[1]*v[2]-x[2]*v[1];
	omega[1]=x[2]*v[0]-x[0]*v[2];
	omega[2]=x[0]*v[1]-x[1]*v[0];
	om=len(omega)/(r*r);
	
	// Potential derivative
	delta=0.02*kpc;
	for(i=0;i<3;i++){
		x1[i]=x[i]/r*(r-delta);
		x2[i]=x[i]/r*(r+delta);
	}
	force(x1, a1, par, potential);
	force(x2, a2, par, potential);
	for(i=0;i<3;i++){
		dx[i]=x1[i]-x2[i];
	}
	dpot=(len(a1)-len(a2))/len(dx);
	
	// Jacobi radius
	R=pow(G*Mcl/fabs(om*om+dpot),1./3.);
	
	return R;
}
