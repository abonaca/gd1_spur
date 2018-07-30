#include "interact.h"

// Utility functions
double len(double *x)
{	// Length of a vector
	return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

void n2t(double *x, double *x1, double *x2, double *x3, int i)
{	// Fill array x[3] with values at xj[i], j=1-3
	x[0]=x1[i];
	x[1]=x2[i];
	x[2]=x3[i];
}

void t2n(double *x, double *x1, double *x2, double *x3, int i)
{	// Fill arrays xj[i] with values from x[3], j=1-3
	x1[i]=x[0];
	x2[i]=x[1];
	x3[i]=x[2];
}

void t2t(double *x, double *y)
{	// Copy array x to array y
	y[0]=x[0];
	y[1]=x[1];
	y[2]=x[2];
}
