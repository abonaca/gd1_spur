#include <Python.h>
#include <numpy/arrayobject.h>
#include "interact.h"

/* Docstrings */
static char module_docstring[] = "This module provides an interface for running a leapfrog orbit integrator using C.";
static char stream_docstring[] = "Calculate positions and velocities of a streakline stream, given the progenitor initial position and gravitational potential.";
static char orbit_docstring[] = "Calculate orbit in a parameterized potential given the inital position and velocity.";
static char encounter_docstring[] = "Calculate orbit of a perturber";
static char interact_docstring[] = "Calculate reactions of a tube of stars to a perturber flyby";
static char general_interact_docstring[] = "Calculate reactions of a tube of stars to a generalized perturber flyby";
    
/* Available functions */
static PyObject *interact_stream(PyObject *self, PyObject *args);
static PyObject *interact_orbit(PyObject *self, PyObject *args);
static PyObject *interact_encounter(PyObject *self, PyObject *args);
static PyObject *interact_interact(PyObject *self, PyObject *args);
static PyObject *interact_general_interact(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
	{"stream", interact_stream, METH_VARARGS, stream_docstring},
	{"orbit", interact_orbit, METH_VARARGS, orbit_docstring},
	{"encounter", interact_encounter, METH_VARARGS, encounter_docstring},
	{"interact", interact_interact, METH_VARARGS, interact_docstring},
	{"general_interact", interact_general_interact, METH_VARARGS, general_interact_docstring},
	{NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initinteract(void)
{
	PyObject *m = Py_InitModule3("interact", module_methods, module_docstring);
	if (m == NULL)
		return;

	/* Load `numpy` functionality. */
	import_array();
}

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);


static PyObject *interact_general_interact(PyObject *self, PyObject *args)
{
    // Parse the input tuple
    int potential, potential_perturb;
    double Tenc, T, dt_;
    PyObject *par_perturb_obj, *par_perturb_array, *x0_obj, *x0_array, *v0_obj, *v0_array, *par_pot_obj, *par_pot_array, *x1_obj, *x1_array, *x2_obj, *x2_array, *x3_obj, *x3_array, *v1_obj, *v1_array, *v2_obj, *v2_array, *v3_obj, *v3_array;

    // reads in input parameters
    if (!PyArg_ParseTuple(args, "OOOdddOiiOOOOOO", &par_perturb_obj, &x0_obj, &v0_obj, &Tenc, &T, &dt_, &par_pot_obj, &potential, &potential_perturb, &x1_obj, &x2_obj, &x3_obj, &v1_obj, &v2_obj, &v3_obj))
        return NULL;

    // Interpret the input parameters as numpy arrays
    par_perturb_array = PyArray_FROM_OTF(par_perturb_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x0_array = PyArray_FROM_OTF(x0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v0_array = PyArray_FROM_OTF(v0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    par_pot_array = PyArray_FROM_OTF(par_pot_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x1_array = PyArray_FROM_OTF(x1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x2_array = PyArray_FROM_OTF(x2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x3_array = PyArray_FROM_OTF(x3_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v1_array = PyArray_FROM_OTF(v1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v2_array = PyArray_FROM_OTF(v2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v3_array = PyArray_FROM_OTF(v3_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    //If that didn't work, throw an exception
    if (par_perturb_array == NULL) {
        Py_XDECREF(par_perturb_array);
        return NULL;
    }
    if (x0_array == NULL) {
        Py_XDECREF(x0_array);
        return NULL;
    }
    if (v0_array == NULL) {
        Py_XDECREF(v0_array);
        return NULL;
    }
    if (par_pot_array == NULL) {
        Py_XDECREF(par_pot_array);
        return NULL;
    }
    if (x1_array == NULL) {
        Py_XDECREF(x1_array);
        return NULL;
    }
    if (x2_array == NULL) {
        Py_XDECREF(x2_array);
        return NULL;
    }
    if (x3_array == NULL) {
        Py_XDECREF(x3_array);
        return NULL;
    }
    if (v1_array == NULL) {
        Py_XDECREF(v1_array);
        return NULL;
    }
    if (v2_array == NULL) {
        Py_XDECREF(v2_array);
        return NULL;
    }
    if (v3_array == NULL) {
        Py_XDECREF(v3_array);
        return NULL;
    }

    // How many stars are there?
    int Nstar = (int)PyArray_DIM(x1_array, 0);

    // Get pointers to the data as C-types
    double *par_perturb, *x0, *v0, *par_pot, *x1, *x2, *x3, *v1, *v2, *v3;
    par_perturb = (double*)PyArray_DATA(par_perturb_array);
    x0 = (double*)PyArray_DATA(x0_array);
    v0 = (double*)PyArray_DATA(v0_array);
    par_pot = (double*)PyArray_DATA(par_pot_array);
    x1 = (double*)PyArray_DATA(x1_array);
    x2 = (double*)PyArray_DATA(x2_array);
    x3 = (double*)PyArray_DATA(x3_array);
    v1 = (double*)PyArray_DATA(v1_array);
    v2 = (double*)PyArray_DATA(v2_array);
    v3 = (double*)PyArray_DATA(v3_array);
    
	// Call the external C function to calculate the interaction
	int err; 
    err = general_interact(par_perturb, x0, v0, Tenc, T, dt_, par_pot, potential, potential_perturb, Nstar, x1, x2, x3, v1, v2, v3);
    
	// Check if error raised
	if(err!=0) {
		PyErr_SetString(PyExc_RuntimeError, "Error occured in the leapfrog integrator.");
		return NULL;
	}
	
	// Store return array
	PyObject *out = Py_BuildValue("OOOOOO", x1_array, x2_array, x3_array, v1_array, v2_array, v3_array);
	
	// Return positions, velocities and energy as a function of time
	return out;
}

static PyObject *interact_interact(PyObject *self, PyObject *args)
{
    // Parse the input tuple
    int potential, potential_perturb;
    double B, phi, V, theta, Tenc, T, dt_;
    PyObject *par_perturb_obj, *par_perturb_array, *par_pot_obj, *par_pot_array, *x1_obj, *x1_array, *x2_obj, *x2_array, *x3_obj, *x3_array, *v1_obj, *v1_array, *v2_obj, *v2_array, *v3_obj, *v3_array;

    // reads in input parameters
    if (!PyArg_ParseTuple(args, "OdddddddOiiOOOOOO", &par_perturb_obj, &B, &phi, &V, &theta, &Tenc, &T, &dt_, &par_pot_obj, &potential, &potential_perturb, &x1_obj, &x2_obj, &x3_obj, &v1_obj, &v2_obj, &v3_obj))
        return NULL;

    // Interpret the input parameters as numpy arrays
    par_perturb_array = PyArray_FROM_OTF(par_perturb_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    par_pot_array = PyArray_FROM_OTF(par_pot_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x1_array = PyArray_FROM_OTF(x1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x2_array = PyArray_FROM_OTF(x2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    x3_array = PyArray_FROM_OTF(x3_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v1_array = PyArray_FROM_OTF(v1_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v2_array = PyArray_FROM_OTF(v2_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    v3_array = PyArray_FROM_OTF(v3_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    //If that didn't work, throw an exception
    if (par_perturb_array == NULL) {
        Py_XDECREF(par_perturb_array);
        return NULL;
    }
    if (par_pot_array == NULL) {
        Py_XDECREF(par_pot_array);
        return NULL;
    }
    if (x1_array == NULL) {
        Py_XDECREF(x1_array);
        return NULL;
    }
    if (x2_array == NULL) {
        Py_XDECREF(x2_array);
        return NULL;
    }
    if (x3_array == NULL) {
        Py_XDECREF(x3_array);
        return NULL;
    }
    if (v1_array == NULL) {
        Py_XDECREF(v1_array);
        return NULL;
    }
    if (v2_array == NULL) {
        Py_XDECREF(v2_array);
        return NULL;
    }
    if (v3_array == NULL) {
        Py_XDECREF(v3_array);
        return NULL;
    }

    // How many stars are there?
    int Nstar = (int)PyArray_DIM(x1_array, 0);

    // Get pointers to the data as C-types
    double *par_perturb, *par_pot, *x1, *x2, *x3, *v1, *v2, *v3;
    par_perturb = (double*)PyArray_DATA(par_perturb_array);
    par_pot = (double*)PyArray_DATA(par_pot_array);
    x1 = (double*)PyArray_DATA(x1_array);
    x2 = (double*)PyArray_DATA(x2_array);
    x3 = (double*)PyArray_DATA(x3_array);
    v1 = (double*)PyArray_DATA(v1_array);
    v2 = (double*)PyArray_DATA(v2_array);
    v3 = (double*)PyArray_DATA(v3_array);
    
	// Call the external C function to calculate the interaction
	int err; 
    err = interact(par_perturb, B, phi, V, theta, Tenc, T, dt_, par_pot, potential, potential_perturb, Nstar, x1, x2, x3, v1, v2, v3);
    
	// Check if error raised
	if(err!=0) {
		PyErr_SetString(PyExc_RuntimeError, "Error occured in the leapfrog integrator.");
		return NULL;
	}
	
	// Store return array
	PyObject *out = Py_BuildValue("OOOOOO", x1_array, x2_array, x3_array, v1_array, v2_array, v3_array);
	
	// Return positions, velocities and energy as a function of time
	return out;
}

static PyObject *interact_encounter(PyObject *self, PyObject *args)
{
// 	int N, err, potential, integrator;
// 	double *x0, *v0, *par, dt_, direction;
// 	PyObject *par_obj, *par_array, *x0_obj, *x0_array, *v0_obj, *v0_array;

	// Parse the input tuple
    double M, B, phi, V, theta, T, dt_;
//     int encounter(double M, double B, double phi, double V, double theta, double T, double dt_, double *x1, double *x2, double *x3, double *v1, double *v2, double *v3)
    if (!PyArg_ParseTuple(args, "ddddddd", &M, &B, &phi, &V, &theta, &T, &dt_))	// reads in input parameters
		return NULL;
// 	if (!PyArg_ParseTuple(args, "OOOiiidd", &x0_obj, &v0_obj, &par_obj, &potential, &integrator, &N, &dt_, &direction))	// reads in input parameters
// 		return NULL;
// 	Ne=ceil((float)N/(float)M);
	
// 	// Interpret the input parameters as numpy arrays
// 	par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_IN_ARRAY);
// 	x0_array = PyArray_FROM_OTF(x0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
// 	v0_array = PyArray_FROM_OTF(v0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
// 	
// 	//If that didn't work, throw an exception
// 	if (par_array == NULL) {
// 		Py_XDECREF(par_array);
// 		return NULL;
// 	}
// 	if (x0_array == NULL) {
// 		Py_XDECREF(x0_array);
// 		return NULL;
// 	}
// 	if (v0_array == NULL) {
// 		Py_XDECREF(v0_array);
// 		return NULL;
// 	}
// 	// How many parameters are there?
// // 	int Npar = (int)PyArray_DIM(par_array, 0);
// 
// 	//Get pointers to the data as C-types. */
// 	par = (double*)PyArray_DATA(par_array);
// 	x0 = (double*)PyArray_DATA(x0_array);
// 	v0 = (double*)PyArray_DATA(v0_array);
	
	// Set up return array pointers
	double *x1, *x2, *x3, *v1, *v2, *v3;
	int err, nd=1;
	npy_intp dims[2];
    int N = T/dt_;
	dims[0] = 2*N + 1;
	PyArrayObject *py_x1, *py_x2, *py_x3, *py_v1, *py_v2, *py_v3;
	
	// Python arrays
	py_x1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_x2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_x3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_v1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_v2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_v3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	
	// Pointers to C arrays
	x1 = pyvector_to_Carrayptrs(py_x1);
	x2 = pyvector_to_Carrayptrs(py_x2);
	x3 = pyvector_to_Carrayptrs(py_x3);
	v1 = pyvector_to_Carrayptrs(py_v1);
	v2 = pyvector_to_Carrayptrs(py_v2);
	v3 = pyvector_to_Carrayptrs(py_v3);

	// Call the external C function to calculate the geostationary orbit.
// 	printf("before %e\n", x1[0]);
	err = encounter(M, B, phi, V, theta, T, dt_, x1, x2, x3, v1, v2, v3);
// 	printf("after %e\n", x1[0]);

	// Check if error raised
	if(err!=0) {
		PyErr_SetString(PyExc_RuntimeError, "Error occured in the leapfrog integrator.");
		return NULL;
	}
	
	// Store return array
	PyObject *out = Py_BuildValue("OOOOOO", py_x1, py_x2, py_x3, py_v1, py_v2, py_v3);
	
	// Clean up
// 	Py_XDECREF(par_array);
	Py_XDECREF(py_x1);
	Py_XDECREF(py_x2);
	Py_XDECREF(py_x3);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_v3);
	
	// Return positions, velocities and energy as a function of time
	return out;
}


static PyObject *interact_stream(PyObject *self, PyObject *args)
{
	int N, M, Ne, err, potential, integrator;
	double *x0, *v0, *par, *offset, mcli, mclf, rcl, dt_;
	PyObject *par_obj, *par_array, *x0_obj, *x0_array, *v0_obj, *v0_array, *offset_obj, *offset_array;

	// Parse the input tuple
	if (!PyArg_ParseTuple(args, "OOOOiiiidddd", &x0_obj, &v0_obj, &par_obj, &offset_obj, &potential, &integrator, &N, &M, &mcli, &mclf, &rcl, &dt_))	// reads in input parameters: initial position and velocity, choice of potential, potential parameters and the number of timesteps N
		return NULL;
	Ne=ceil((float)N/(float)M);
	
	// Interpret the input parameters as numpy arrays
	par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	offset_array = PyArray_FROM_OTF(offset_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	x0_array = PyArray_FROM_OTF(x0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	v0_array = PyArray_FROM_OTF(v0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	
	//If that didn't work, throw an exception
	if (par_array == NULL) {
		Py_XDECREF(par_array);
		return NULL;
	}
	if (offset_array == NULL) {
		Py_XDECREF(offset_array);
		return NULL;
	}
	if (x0_array == NULL) {
		Py_XDECREF(x0_array);
		return NULL;
	}
	if (v0_array == NULL) {
		Py_XDECREF(v0_array);
		return NULL;
	}
	// How many parameters are there?
// 	int Npar = (int)PyArray_DIM(par_array, 0);

	//Get pointers to the data as C-types. */
	par = (double*)PyArray_DATA(par_array);
	offset = (double*)PyArray_DATA(offset_array);
	x0 = (double*)PyArray_DATA(x0_array);
	v0 = (double*)PyArray_DATA(v0_array);
	
	// Set up return array pointers
	double *xm1, *xm2, *xm3, *xp1, *xp2, *xp3, *vm1, *vm2, *vm3, *vp1, *vp2, *vp3;
	int nd=1;
// 	npy_intp *dims;
// 	dims[0] = Ne;
	npy_intp dims[2];
	dims[0] = Ne;
	PyArrayObject *py_xm1, *py_xm2, *py_xm3, *py_xp1, *py_xp2, *py_xp3, *py_vm1, *py_vm2, *py_vm3, *py_vp1, *py_vp2, *py_vp3;
	
	// Python arrays
	py_xm1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_xm2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_xm3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_xp1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_xp2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_xp3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_vm1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_vm2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_vm3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_vp1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_vp2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_vp3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	
	// Pointers to C arrays
	xm1 = pyvector_to_Carrayptrs(py_xm1);
	xm2 = pyvector_to_Carrayptrs(py_xm2);
	xm3 = pyvector_to_Carrayptrs(py_xm3);
	xp1 = pyvector_to_Carrayptrs(py_xp1);
	xp2 = pyvector_to_Carrayptrs(py_xp2);
	xp3 = pyvector_to_Carrayptrs(py_xp3);
	vm1 = pyvector_to_Carrayptrs(py_vm1);
	vm2 = pyvector_to_Carrayptrs(py_vm2);
	vm3 = pyvector_to_Carrayptrs(py_vm3);
	vp1 = pyvector_to_Carrayptrs(py_vp1);
	vp2 = pyvector_to_Carrayptrs(py_vp2);
	vp3 = pyvector_to_Carrayptrs(py_vp3);

	// Call the external C function to calculate the geostationary orbit.
	err = stream(x0, v0, xm1, xm2, xm3, xp1, xp2, xp3, vm1, vm2, vm3, vp1, vp2, vp3, par, offset, potential, integrator, N, M, mcli, mclf, rcl, dt_);

	// Check if error raised
	if(err!=0) {
		PyErr_SetString(PyExc_RuntimeError, "Error occured in the leapfrog integrator.");
		return NULL;
	}
	
	// Store return array
	PyObject *out = Py_BuildValue("OOOOOOOOOOOO", py_xm1, py_xm2, py_xm3, py_xp1, py_xp2, py_xp3, py_vm1, py_vm2, py_vm3, py_vp1, py_vp2, py_vp3);
	
	// Clean up
	Py_XDECREF(par_array);
	Py_XDECREF(py_xm1);
	Py_XDECREF(py_xm2);
	Py_XDECREF(py_xm3);
	Py_XDECREF(py_xp1);
	Py_XDECREF(py_xp2);
	Py_XDECREF(py_xp3);
	Py_XDECREF(py_vm1);
	Py_XDECREF(py_vm2);
	Py_XDECREF(py_vm3);
	Py_XDECREF(py_vp1);
	Py_XDECREF(py_vp2);
	Py_XDECREF(py_vp3);
	
	// Return positions, velocities and energy as a function of time
	return out;
}

static PyObject *interact_orbit(PyObject *self, PyObject *args)
{
	int N, err, potential, integrator;
	double *x0, *v0, *par, dt_, direction;
	PyObject *par_obj, *par_array, *x0_obj, *x0_array, *v0_obj, *v0_array;

	// Parse the input tuple
	if (!PyArg_ParseTuple(args, "OOOiiidd", &x0_obj, &v0_obj, &par_obj, &potential, &integrator, &N, &dt_, &direction))	// reads in input parameters
		return NULL;
// 	Ne=ceil((float)N/(float)M);
	
	// Interpret the input parameters as numpy arrays
	par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	x0_array = PyArray_FROM_OTF(x0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	v0_array = PyArray_FROM_OTF(v0_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	
	//If that didn't work, throw an exception
	if (par_array == NULL) {
		Py_XDECREF(par_array);
		return NULL;
	}
	if (x0_array == NULL) {
		Py_XDECREF(x0_array);
		return NULL;
	}
	if (v0_array == NULL) {
		Py_XDECREF(v0_array);
		return NULL;
	}
	// How many parameters are there?
// 	int Npar = (int)PyArray_DIM(par_array, 0);

	//Get pointers to the data as C-types. */
	par = (double*)PyArray_DATA(par_array);
	x0 = (double*)PyArray_DATA(x0_array);
	v0 = (double*)PyArray_DATA(v0_array);
	
	// Set up return array pointers
	double *x1, *x2, *x3, *v1, *v2, *v3;
	int nd=1;
// 	npy_intp *dims;
// 	dims[0] = Ne;
	npy_intp dims[2];
	dims[0] = N;
	PyArrayObject *py_x1, *py_x2, *py_x3, *py_v1, *py_v2, *py_v3;
	
	// Python arrays
	py_x1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_x2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_x3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_v1 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_v2 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	py_v3 = (PyArrayObject*) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	
	// Pointers to C arrays
	x1 = pyvector_to_Carrayptrs(py_x1);
	x2 = pyvector_to_Carrayptrs(py_x2);
	x3 = pyvector_to_Carrayptrs(py_x3);
	v1 = pyvector_to_Carrayptrs(py_v1);
	v2 = pyvector_to_Carrayptrs(py_v2);
	v3 = pyvector_to_Carrayptrs(py_v3);

	// Call the external C function to calculate the geostationary orbit.
// 	printf("before %e\n", x1[0]);
	err = orbit(x0, v0, x1, x2, x3, v1, v2, v3, par, potential, integrator, N, dt_, direction);
// 	printf("after %e\n", x1[0]);

	// Check if error raised
	if(err!=0) {
		PyErr_SetString(PyExc_RuntimeError, "Error occured in the leapfrog integrator.");
		return NULL;
	}
	
	// Store return array
	PyObject *out = Py_BuildValue("OOOOOO", py_x1, py_x2, py_x3, py_v1, py_v2, py_v3);
	
	// Clean up
	Py_XDECREF(par_array);
	Py_XDECREF(py_x1);
	Py_XDECREF(py_x2);
	Py_XDECREF(py_x3);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_v3);
	
	// Return positions, velocities and energy as a function of time
	return out;
}

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)
{
// 	int n=arrayin->dimensions[0];
	return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
