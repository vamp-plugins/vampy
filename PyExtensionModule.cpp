#include <Python.h>
#include "PyExtensionModule.h"
#include "PyRealTime.h"
#include "PyFeature.h"
#include "PyFeatureSet.h"
#include "PyParameterDescriptor.h"
#include "PyOutputDescriptor.h"
#include "vamp-sdk/Plugin.h"

using namespace std;
using namespace Vamp;
using Vamp::Plugin;
using Vamp::RealTime;

/* Simple Example Function */

static int five=5;

PyObject*
get_five(PyObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ":five")) return NULL;
    return Py_BuildValue("i", five);
}


/*			 Functions Exposed by Vampy 					*/

/*			 Creating PyRealTime Objects 					*/


/* New RealTime object from Frame (with given samplerate) */
static PyObject *
RealTime_frame2RealTime(PyObject *ignored, PyObject *args)
{
	long frame;
	unsigned int sampleRate;

    if (!PyArg_ParseTuple(args, "lI:realtime.frame2RealTime ", 
	&frame,&sampleRate))
		return NULL;

	RealTimeObject *self;
	self = PyObject_New(RealTimeObject, &RealTime_Type); 
	if (self == NULL) return NULL;

	self->rt = new RealTime::RealTime(
	RealTime::frame2RealTime(frame,sampleRate));

	return (PyObject *) self;
}

/*			 Creating PyParameterDescriptor Objects 			  	*/

/* New ParameterDescriptor object 
static PyObject *
ParameterDescriptor_new(PyObject *ignored, PyObject *args)
{

    if (!PyArg_ParseTuple(args, ":ParameterDescriptor")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: ParameterDescriptor initialised with arguments.");
		return NULL; 
	  }

	ParameterDescriptorObject *self = 
	PyObject_New(ParameterDescriptorObject, &ParameterDescriptor_Type); 
	if (self == NULL) return NULL;
    self->dict = PyDict_New();
	if (self->dict == NULL) return NULL;
	return (PyObject *) self;
}
*/

/*			 Creating PyOutputDescriptor Objects 					*/

/* New OutputDescriptor object 
static PyObject *
OutputDescriptor_new(PyObject *ignored, PyObject *args)
{

	if (!PyArg_ParseTuple(args, ":OutputDescriptor")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: OutputDescriptor initialised with arguments.");
		return NULL; 
	  }

	OutputDescriptorObject *self = 
	PyObject_New(OutputDescriptorObject, &OutputDescriptor_Type); 
	if (self == NULL) return NULL;
    self->dict = PyDict_New();
	if (self->dict == NULL) return NULL;
	return (PyObject *) self;
}
*/

/*			 Creating PyOutputList Objects 					*/

/* New OutputList object */
static PyObject *
OutputList_new(PyObject *ignored, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ":OutputList")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: OutputList initialised with arguments.");
		return NULL; 
	  }

	return (PyObject *) PyList_New(0);
}


/*			 Creating PyParameterList Objects 					*/

/* New ParameterList object */
static PyObject *
ParameterList_new(PyObject *ignored, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ":ParameterList")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: ParameterList initialised with arguments.");
		return NULL; 
	  }
	return (PyObject *) PyList_New(0);
}


/*			 Creating PyFeatureList Objects 					*/

/* New FeatureList object 
static PyObject *
FeatureList_new(PyObject *ignored, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ":FeatureList")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: FeatureList initialised with arguments.");
		return NULL; 
	  }
	return (PyObject *) PyList_New(0);
}
*/

/*			 Creating PyFeatureSet Objects 					*/

/* New FeatureSet object 
static PyObject *
FeatureSet_new(PyObject *ignored, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ":FeatureSet")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: FeatureSet initialised with arguments.");
		return NULL; 
	  }
	return (PyObject *) PyDict_New();
}
*/


/*		 	Declare the methods exposed by the vampy module 		*/


PyMethodDef VampyMethods[] = {
/*NOTE: This is conventionally static, but limiting the scope
	here will cause seg fault if the declared functions are 
	called back from a Python function wrapped in a C++ class.*/
    {"five", get_five, METH_VARARGS, "Return a number."},

	{"frame2RealTime",	(PyCFunction)RealTime_frame2RealTime,	METH_VARARGS,
		PyDoc_STR("frame2RealTime((int64)frame, (uint32)sampleRate ) -> returns new RealTime object from frame.")},

	/*{"RealTime",	RealTime_new,		METH_VARARGS,
		PyDoc_STR("RealTime() -> returns new RealTime object")},*/

	/*{"Feature",	Feature_new,		METH_VARARGS,
		PyDoc_STR("Feature() -> returns new Feature object")},*/

	/*{"ParameterDescriptor",	ParameterDescriptor_new,		METH_VARARGS,
		PyDoc_STR("ParameterDescriptor() -> returns new ParameterDescriptor object")},

	{"OutputDescriptor",	OutputDescriptor_new,		METH_VARARGS,
		PyDoc_STR("OutputDescriptor() -> returns new OutputDescriptor object")},

	{"FeatureList",	FeatureList_new,		METH_VARARGS,
		PyDoc_STR("FeatureList() -> returns new FeatureList object")},*/

	{"OutputList",	OutputList_new,		METH_VARARGS,
		PyDoc_STR("OutputList() -> returns new OutputList object")},

	{"ParameterList",	ParameterList_new,		METH_VARARGS,
		PyDoc_STR("ParameterList() -> returns new ParameterList object")},

    {NULL, NULL, 0, NULL} 
};

/* Module Documentation */
// PyDoc_STRVAR(vampy_doc,"This module exposes Vamp plugin data type wrappers.");

PyMODINIT_FUNC
initvampy(void)
{
	PyObject* module;

	// if (PyType_Ready(&Feature_Type) < 0) return;
	/// Why do we get a segfault if this is initialised here?
	/*PyType_Ready adds these object to the GC.
	This is OK for an extension module, but it is a mistake here, 
	because the reference count will be decremented in the Vamp 
	wrapper plugin outside the interpreter.
	When the GC tries to visit a deallocated object, it will throw up.*/
	
	RealTime_Type.ob_type = &PyType_Type;
	Feature_Type.ob_type = &PyType_Type;
	OutputDescriptor_Type.ob_type = &PyType_Type;
	ParameterDescriptor_Type.ob_type = &PyType_Type;
	initFeatureSetType(); /// this is derived from the builtin dict

	PyImport_AddModule("vampy");
	module = Py_InitModule("vampy", VampyMethods);
	if (!module) return;
	
	Py_INCREF(&RealTime_Type);
    PyModule_AddObject(module,"RealTime",(PyObject*)&RealTime_Type);
	// Py_INCREF(&RealTime_Type);

	Py_INCREF((PyObject*)&Feature_Type);
	PyModule_AddObject(module,"Feature",(PyObject*)&Feature_Type);
	// Py_INCREF((PyObject*)&Feature_Type);

	Py_INCREF((PyObject*)&FeatureSet_Type);
	PyModule_AddObject(module,"FeatureSet",(PyObject*)&FeatureSet_Type);
	// Py_INCREF((PyObject*)&FeatureSet_Type);

	Py_INCREF((PyObject*)&OutputDescriptor_Type);
	PyModule_AddObject(module,"OutputDescriptor",(PyObject*)&OutputDescriptor_Type);

	Py_INCREF((PyObject*)&ParameterDescriptor_Type);
	PyModule_AddObject(module,"ParameterDescriptor",(PyObject*)&ParameterDescriptor_Type);
	
	cerr << "Vampy: extension module initialised." << endl;
}

/*
NOTE: Why do we need to clean up the module?

The module exposed by Vampy to the embedded interpreter
contains callback functions. These functions are accessed
via function pointers stored in the extension module dictionary.

Unfortunately, when the Vampy shared library is unloaded and
reloaded again during a host session, these addresses might
change. Therefore, we reinitialise the module dict before 
each use. However, this will cause garbage collection errors
or segmentation faults, when elements of the dict of the 
previous session are attempted to free. Therefore, we hold
a global reference count to all initialised Vampy plugins,
and when this reaches zero, we clean up the module dict.

This is an attempt to catch the moment when the shared lib
is finally unloaded and the references are still point to valid
memory addresses.

Why doesn't the GC clean this up correctly?

In a normal Python session the GC would deallocate the module
dict at the end. In embedded python, although the GC appears
to be called when the shared lib is unloaded, the interpreter
is reused. Since there is no C/API call to unload modules,
and at the time of unloading vampy the wrapped function pointers
are still valid, the GC doesn't collect them, nor are they freed
by the interpreter. When vampy is reloaded however, the module
dict will contain invalid addresses. The above procedure solves
this problem.


*/

void cleanModule(void)
{
	PyObject *m = PyImport_AddModule("vampy");
	if (!m) cerr << "Destr: PyImport_AddModule returned NULL!" << endl;
	else {
		// cerr << "Destr: Add module found existing." << endl;
		PyObject *dict = PyModule_GetDict(m);
		Py_ssize_t ln = PyDict_Size(dict);
		cerr << "Destr: Size of module dict = " << (int) ln << endl;
		/// Clean the module dictionary.
		PyDict_Clear(dict);
	    ln = PyDict_Size(dict);
		cerr << "Destr: Cleaned size of module dict = " << (int) ln << endl;
	}
}

