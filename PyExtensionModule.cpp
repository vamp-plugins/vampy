/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "PyExtensionModule.h"
#include "PyRealTime.h"
#include "PyFeature.h"
#include "PyFeatureSet.h"
#include "PyParameterDescriptor.h"
#include "PyOutputDescriptor.h"
#include "vamp/vamp.h"
#include "vamp-sdk/Plugin.h"

using namespace std;
using namespace Vamp;
using Vamp::Plugin;
using Vamp::RealTime;

/*			 Functions Exposed by Vampy 					*/

/*			 Creating PyRealTime Objects from frame count	*/

/* New RealTime object from Frame (with given samplerate)   */
static PyObject *
RealTime_frame2RealTime(PyObject *ignored, PyObject *args)
{
	long frame;
	unsigned int sampleRate;

	if (!(args && PyTuple_GET_SIZE(args) == 2)) {
		PyErr_SetString(PyExc_ValueError,"frame2RealTime requires two arguments: frame and sample rate.");
		return NULL;
	}

	PyObject* pyFrame = PyTuple_GET_ITEM(args,0);
	PyObject* pySampleRate = PyTuple_GET_ITEM(args,1);

	/// frame 
	if (PyInt_Check(pyFrame)) frame = PyInt_AS_LONG(pyFrame);
	else if (PyLong_Check(pyFrame)) frame = PyLong_AsLong(pyFrame);
	else {
		PyErr_SetString(PyExc_ValueError,"frame2RealTime 'frame' argument must be long integer.");
		return NULL;
	}

	/// sample rate
	if (PyInt_Check(pySampleRate)) 
		sampleRate = _long2uint(PyInt_AS_LONG(pySampleRate));
	else if (PyFloat_Check(pySampleRate)) 
		sampleRate = _dbl2uint(PyFloat_AS_DOUBLE(pySampleRate));
	else if (PyLong_Check(pySampleRate)) 
		sampleRate = _long2uint(PyLong_AsLong(pySampleRate));
	else {
		PyErr_SetString(PyExc_ValueError,"frame2RealTime 'sample rate' argument must be int, long or float.");
		return NULL;
	}
	
	if (!sampleRate) {
		PyErr_SetString(PyExc_ValueError,"frame2RealTime 'sample rate' argument overflow error. Argument must be 0 < arg < UINT_MAX.");
		cerr << "Value: " << sampleRate << endl;
		return NULL;
	}
	
	// simpler but slower:
	// if (!PyArg_ParseTuple(args, "lI:realtime.frame2RealTime ", 
	// &frame,&sampleRate))
	// return NULL;

	RealTimeObject *self;
	self = PyObject_New(RealTimeObject, &RealTime_Type); 
	if (self == NULL) return NULL;

	self->rt = new RealTime(
	RealTime::frame2RealTime(frame,sampleRate));

	return (PyObject *) self;
}

/*

Note: these functions are not very interesting on their own, but
they can be used to make the semantics of the plugin clearer.
They return ordinary Python list objects. All type checking 
is performed in the type interface.

*/

/* New PyOutputList Objects */
static PyObject *
OutputList_new(PyObject *ignored, PyObject *args)
{
	if (args && PyTuple_Check(args)) 
		return PySequence_List(args);
	else return (PyObject *) PyList_New(0);
}


/* New PyParameterList Objects */
static PyObject *
ParameterList_new(PyObject *ignored, PyObject *args)
{
	if (args && PyTuple_Check(args)) 
		return PySequence_List(args);
	else return (PyObject *) PyList_New(0);
}

/* New PyFeatureList Objects */
static PyObject *
FeatureList_new(PyObject *ignored, PyObject *args)
{
	if (args && PyTuple_Check(args)) 
		return PySequence_List(args);
	else return (PyObject *) PyList_New(0);
}


/*		 	Declare the methods exposed by the vampy module 		*/


PyMethodDef VampyMethods[] = {
/*NOTE: This is conventionally static, but limiting the scope
	here will cause seg fault if the declared functions are 
	called back from a Python function wrapped in a C++ class.*/

	{"frame2RealTime",	(PyCFunction)RealTime_frame2RealTime,	METH_VARARGS,
		PyDoc_STR("frame2RealTime((int64)frame, (uint32)sampleRate ) -> returns new RealTime object from frame.")},

	{"OutputList",	OutputList_new,		METH_VARARGS,
		PyDoc_STR("OutputList() -> returns new OutputList object")},

	{"ParameterList",	ParameterList_new,		METH_VARARGS,
		PyDoc_STR("ParameterList() -> returns new ParameterList object")},

	{"FeatureList",	FeatureList_new,		METH_VARARGS,
		PyDoc_STR("FeatureList() -> returns new FeatureList object")},

    {NULL, NULL, 0, NULL} 
};

/* Module Documentation */
// PyDoc_STRVAR(vampy_doc,"This module exposes Vamp plugin data type wrappers.");

static int
setint(PyObject *d, char *name, int value)
{
	PyObject *v;
	int err;
	v = PyInt_FromLong((long)value);
	err = PyDict_SetItemString(d, name, v);
	Py_XDECREF(v);
	return err;
}

static int
setdbl(PyObject *d, char *name, double value)
{
	PyObject *v;
	int err;
	v = PyFloat_FromDouble(value);
	err = PyDict_SetItemString(d, name, v);
	Py_XDECREF(v);
	return err;
}

static int
setstr(PyObject *d, char *name, char *value)
{
	PyObject *v;
	int err;
	v = PyString_FromString(value);
	err = PyDict_SetItemString(d, name, v);
	Py_XDECREF(v);
	return err;
}


PyMODINIT_FUNC
initvampy(void)
{
	PyObject *module, *mdict;

	/* if (PyType_Ready(&Feature_Type) < 0) return;
	Note: Why do we get a segfault if this is initialised here?
	PyType_Ready adds these object to the GC.
	This is OK for an extension module, but it is a mistake here, 
	because the adresses become invalid when the shared library
	is unloaded. When the GC tries to visit a these objects, 
	it will fail.*/
	
	RealTime_Type.ob_type = &PyType_Type;
	Feature_Type.ob_type = &PyType_Type;
	OutputDescriptor_Type.ob_type = &PyType_Type;
	ParameterDescriptor_Type.ob_type = &PyType_Type;
	initFeatureSetType(); // this is derived from the builtin dict

	PyImport_AddModule("vampy");
	module = Py_InitModule("vampy", VampyMethods);
	if (!module) goto failure;
	mdict = PyModule_GetDict(module);
	if (!mdict) goto failure;

	/// vampy plugin wrapper flags
	if (setint(mdict, "vf_NULL", vf_NULL) < 0) goto failure;
	if (setint(mdict, "vf_DEBUG", vf_DEBUG) < 0) goto failure;
	if (setint(mdict, "vf_STRICT", vf_STRICT) < 0) goto failure;
	if (setint(mdict, "vf_QUIT", vf_QUIT) < 0) goto failure;
	if (setint(mdict, "vf_REALTIME", vf_REALTIME) < 0) goto failure;
	if (setint(mdict, "vf_BUFFER", vf_BUFFER) < 0) goto failure;
	if (setint(mdict, "vf_ARRAY", vf_ARRAY) < 0) goto failure;
	if (setint(mdict, "vf_DEFAULT_V2", vf_DEFAULT_V2) < 0) goto failure;
	
	/// Vamp enum types simulation
	if (setint(mdict, "OneSamplePerStep", Vamp::Plugin::OutputDescriptor::OneSamplePerStep) < 0) goto failure;
	if (setint(mdict, "FixedSampleRate", Vamp::Plugin::OutputDescriptor::FixedSampleRate) < 0) goto failure;
	if (setint(mdict, "VariableSampleRate", Vamp::Plugin::OutputDescriptor::VariableSampleRate) < 0) goto failure;
	if (setint(mdict, "TimeDomain", Vamp::Plugin::TimeDomain) < 0) goto failure;
	if (setint(mdict, "FrequencyDomain", Vamp::Plugin::FrequencyDomain) < 0) goto failure;

	/// module attributes
	if (setstr(mdict, "__name__", "vampy") < 0) goto failure;
	if (setdbl(mdict, "__version__", 2.0) < 0) goto failure;
	if (setdbl(mdict, "__VAMP_API_VERSION__", (double) VAMP_API_VERSION) < 0) goto failure;
#ifdef HAVE_NUMPY
	if (setint(mdict, "__numpy__", 1) < 0) goto failure;
#else
	if (setint(mdict, "__numpy__", 0) < 0) goto failure;
#endif
	
	/// type objects
	Py_INCREF(&RealTime_Type);
	if (PyModule_AddObject(module,"RealTime",(PyObject*)&RealTime_Type) !=0) goto failure;

	Py_INCREF((PyObject*)&Feature_Type);
	if (PyModule_AddObject(module,"Feature",(PyObject*)&Feature_Type) !=0) goto failure;

	Py_INCREF((PyObject*)&FeatureSet_Type);
	if (PyModule_AddObject(module,"FeatureSet",(PyObject*)&FeatureSet_Type) !=0) goto failure;

	Py_INCREF((PyObject*)&OutputDescriptor_Type);
	if (PyModule_AddObject(module,"OutputDescriptor",(PyObject*)&OutputDescriptor_Type) !=0) goto failure;

	Py_INCREF((PyObject*)&ParameterDescriptor_Type);
	if (PyModule_AddObject(module,"ParameterDescriptor",(PyObject*)&ParameterDescriptor_Type) !=0) goto failure;
	
#ifdef _DEBUG	
	cerr << "Vampy: extension module initialised." << endl;
#endif

	return;
	
failure :
	if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
	cerr << "Vampy::PyExtensionModule::initvampy: Failed to initialise extension module." << endl;
	return;
}


