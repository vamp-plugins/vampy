#include <Python.h>
#include "PyExtensionModule.h"
#include "PyFeature.h"
#include "vamp-sdk/Plugin.h"
#include <string>
/*#include "PyTypeInterface.h"*/

using namespace std;
using namespace Vamp;
using Vamp::Plugin;

/*					 Feature Object's Methods 					*/ 
//Feature objects have no callable methods

/*		   PyFeature methods implementing protocols 		   	   */ 
// these functions are called by the interpreter automatically


/* Function to set basic attributes 
static int
Feature_setattr(FeatureObject *self, char *name, PyObject *value)
{
	std::string key = std::string(name);
	if (self->ti.SetValue(*(self->feature),key,value)) return 0;
	else return -1;
}*/

/* Function to get basic attributes 
static PyObject *
Feature_getattr(FeatureObject *self, char *name)
{
	std::string key = std::string(name);
	PyObject* pyValue;
	if (self->ti.GetValue(*(self->feature),key,pyValue)) 
		return pyValue;
	else return NULL;
}*/

/* Set attributes */
static int
Feature_setattr(FeatureObject *self, char *name, PyObject *v)
{
	if (v == NULL) {
		int rv = PyDict_DelItemString(self->dict, name);
		if (rv < 0)
			PyErr_SetString(PyExc_AttributeError,"non-existing Feature attribute");
		return rv;
	}
	else
		return PyDict_SetItemString(self->dict, name, v);
}


/* Get attributes */
static PyObject *
Feature_getattr(FeatureObject *self, char *name)
{
	if (self->dict != NULL) {
		PyObject *v = PyDict_GetItemString(self->dict, name);
		if (v != NULL) 
		{
			Py_INCREF(v);
			return v;
		}
	}
	return NULL;
}

/* New Feature object */
static PyObject *
Feature_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
	/// TODO support kwargs e.g. Feature(values = val, timestamp = ts)
	cerr << "FeatureObject new method called" << endl;
	if (!PyArg_ParseTuple(args, ":Feature")) { 
		PyErr_SetString(PyExc_TypeError, 
		"Error: Feature initialised with arguments.");
		return NULL; 
	}
	FeatureObject *self = (FeatureObject*)type->tp_alloc(type, 0);
	// FeatureObject *self = PyObject_New(FeatureObject, &Feature_Type); 
	if (self == NULL) return NULL;
    self->dict = PyDict_New();
	if (self->dict == NULL) return NULL;
	return (PyObject *) self;
}

/* DESTRUCTOR: delete type object */
static void
FeatureObject_dealloc(FeatureObject *self)
{
	Py_XDECREF(self->dict);
	// delete self->feature; 	//delete the C object
	// PyObject_Del(self); //delete the Python object
	self->ob_type->tp_free((PyObject*)self);
	cerr << "Feature object deallocated." << endl;
}


static int
Feature_init(FeatureObject *self, PyObject *args, PyObject *kwds)
{
	cerr << "FeatureObject Init called" << endl;
	return 0;
}

PyObject*
Feature_test(PyObject *self, PyObject *args, PyObject *kwds)
{
	cerr << "FeatureObject TEST called" << endl;
	return self;
}


/* String representation */
static PyObject *
Feature_repr(PyObject *self)
{
	// if (PyFeature_CheckExact(self)) {}
	// PyObject* intdict = self
	return Py_BuildValue("s",
		"not yet implemented");
	// ((RealTimeObject*)self)->rt->toString().c_str());
}

#define Feature_alloc PyType_GenericAlloc
#define Feature_free PyObject_Del


/*						FEATURE TYPE OBJECT						*/

PyTypeObject Feature_Type = {
	PyObject_HEAD_INIT(NULL)
	0,						/*ob_size*/
	"vampy.Feature",		/*tp_name*/
	sizeof(FeatureObject),	/*tp_basicsize*/
	0,						/*tp_itemsize*/
	(destructor)FeatureObject_dealloc, /*tp_dealloc*/
	0,						/*tp_print*/
	(getattrfunc)Feature_getattr, /*tp_getattr*/
	(setattrfunc)Feature_setattr, /*tp_setattr*/
	0,						/*tp_compare*/
	Feature_repr,			/*tp_repr*/
	0,						/*tp_as_number*/
	0,						/*tp_as_sequence*/
	0,						/*tp_as_mapping*/
	0,						/*tp_hash*/
	Feature_test,           /*tp_call*/ // call on an instance
    0,                      /*tp_str*/
    0,                      /*tp_getattro*/
    0,                      /*tp_setattro*/
    0,                      /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /*tp_flags*/
    0,                      /*tp_doc*/
    0,                      /*tp_traverse*/
    0,                      /*tp_clear*/
    0,                      /*tp_richcompare*/
    0,                      /*tp_weaklistoffset*/
    0,                      /*tp_iter*/
    0,                      /*tp_iternext*/
    0,				        /*tp_methods*/ //TypeObject Methods
    0,                      /*tp_members*/
    0,                      /*tp_getset*/
    0,                      /*tp_base*/
    0,                      /*tp_dict*/
    0,                      /*tp_descr_get*/
    0,                      /*tp_descr_set*/
    0,                      /*tp_dictoffset*/
    0,//(initproc)Feature_init, /*tp_init*/
    Feature_alloc,          /*tp_alloc*/
    Feature_new,            /*tp_new*/
    Feature_free,			/*tp_free*/
    0,                      /*tp_is_gc*/
};

/*		  		 	  PyRealTime C++ API  	  		  				*/

/*Feature* from PyFeature
const Vamp::Plugin::Feature*
PyFeature_AsFeature (PyObject *self) { 

	FeatureObject *s = (FeatureObject*) self; 

	if (!PyFeature_Check(s)) {
		PyErr_SetString(PyExc_TypeError, "Feature Object Expected.");
		cerr << "in call PyFeature_AsPointer(): Feature Object Expected. " << endl;
		return NULL; }
	return s->feature; 
};*/
