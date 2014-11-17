/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "PyExtensionModule.h"
#include "PyFeature.h"
#include "vamp-sdk/Plugin.h"
#include <string>

using namespace std;
using namespace Vamp;
using Vamp::Plugin;

/* CONSTRUCTOR: New Feature object */
static PyObject *
Feature_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
	// FeatureObject *self = PyObject_New(FeatureObject, &Feature_Type); 
	FeatureObject *self = (FeatureObject*)type->tp_alloc(type, 0);
	if (self == NULL) return NULL;
    self->dict = PyDict_New();
	if (self->dict == NULL) return NULL;

	/// 4 args max.: {values|self_copy},timestamp,duration,label 
	if(args && PyTuple_GET_SIZE(args)>0) {
		int s =  PyTuple_GET_SIZE(args);
		PyObject* arg0 = PyTuple_GET_ITEM(args,0);
		if (s == 1 && PyFeature_CheckExact(arg0))
			PyDict_Merge(self->dict,PyFeature_AS_DICT(arg0),0);
		else
			PyDict_SetItemString(self->dict, "values", arg0);
		if (s>1) {
			PyDict_SetItemString(self->dict, "timestamp", PyTuple_GET_ITEM(args,1));
			PyDict_SetItemString(self->dict, "hasTimestamp", Py_True);
		}
		if (s>2) {
			PyDict_SetItemString(self->dict, "duration", PyTuple_GET_ITEM(args,2));
			PyDict_SetItemString(self->dict, "hasDuration", Py_True);
		}
		if (s>3) {
			PyDict_SetItemString(self->dict, "label", PyTuple_GET_ITEM(args,3));
		}
	}

	/// accept keyword arguments: 
	/// e.g. Feature(values = theOutputArray)
	if (!kw || !PyDict_Size(kw)) return (PyObject *) self;
	PyDict_Merge(self->dict,kw,0);

	static const char *kwlist[] = {"timestamp", "hasTimestamp", "duration", "hasDuration", 0};

	int i = 0;
	while (kwlist[i]) {
		const char* name = kwlist[i];
		const char* attr = kwlist[++i];
		i++;
		PyObject *key = PyString_FromString(name);
		if (!key) break;
		if (PyDict_Contains(kw,key)) {
			if (PyDict_SetItem(self->dict,PyString_FromString(attr),Py_True) != 0)
				PyErr_SetString(PyExc_TypeError, 
					"Error: in keyword arguments of vampy.Feature().");
		}
		Py_DECREF(key);
	}

	return (PyObject *) self;
}

/* DESTRUCTOR: delete type object */
static void
FeatureObject_dealloc(FeatureObject *self)
{
	Py_XDECREF(self->dict);
	self->ob_type->tp_free((PyObject*)self);
}

/*					 Feature Object's Methods 					*/ 
//Feature objects have no callable methods

/*		   PyFeature methods implementing protocols 		   	   */ 
// these functions are called by the interpreter automatically

/* Set attributes */
static int
Feature_setattr(FeatureObject *self, char *name, PyObject *v)
{
	if (v == NULL) 
	{
		int rv = PyDict_DelItemString(self->dict, name);
		if (rv < 0)
			PyErr_SetString(PyExc_AttributeError,"non-existing Feature attribute");
		return rv;
	}
	else return PyDict_SetItemString(self->dict, name, v);
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

/* The problem with this is that we'd need to implement two-way
conversions which is really unnecesary: The case for using 
a Vamp::Feature in Python for anything else than returning 
values is rather obscure. It's not really worth it. */

/* Set Attribute: Using wrapped Vamp::Feature 
static int
Feature_setattr(FeatureObject *self, char *name, PyObject *value)
{
	std::string key = std::string(name);
	if (self->ti.SetValue(*(self->feature),key,value)) return 0;
	else return -1;
}*/

/* Get Attribute: Using wrapped Vamp::Feature 
static PyObject *
Feature_getattr(FeatureObject *self, char *name)
{
	std::string key = std::string(name);
	PyObject* pyValue;
	if (self->ti.GetValue(*(self->feature),key,pyValue)) 
		return pyValue;
	else return NULL;
}*/

/*
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
*/

/* String representation */
static PyObject *
Feature_repr(PyObject *self)
{
	FeatureObject* v = (FeatureObject*)self;
	if (v->dict) return PyDict_Type.tp_repr((PyObject *)v->dict);
	else return PyString_FromString("Feature()");
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
	0,//Feature_test,           /*tp_call*/ // call on an instance
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
