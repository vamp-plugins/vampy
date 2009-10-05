/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "PyOutputDescriptor.h"
#include "vamp-sdk/Plugin.h"
#include <string>
#include "PyTypeInterface.h"

using namespace std;
using namespace Vamp;
using Vamp::Plugin;

/*			 OutputDescriptor Object's Methods 					*/ 
//these objects have no callable methods

/*		   PyOutputDescriptor methods implementing protocols   	*/ 
// these functions are called by the interpreter automatically

/* New OutputDescriptor object */ 
static PyObject *
OutputDescriptor_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
	OutputDescriptorObject *self = 
	(OutputDescriptorObject*)type->tp_alloc(type, 0);
	
	if (self == NULL) return NULL;
    self->dict = PyDict_New();
	if (self->dict == NULL) return NULL;
    
	/// allow copying objects
    if (args and PyTuple_Size(args) == 1) {
		PyObject* arg = PyTuple_GET_ITEM(args,0);
		if (PyOutputDescriptor_CheckExact(arg))
			PyDict_Merge(self->dict,PyOutputDescriptor_AS_DICT(arg),0);
		else if (PyDict_CheckExact(arg)) 
			PyDict_Merge(self->dict,arg,0);
		else {
			PyErr_SetString(PyExc_TypeError, 
			"OutputDescriptor takes zero or one PyOutputDescriptor or dictionary arguments.");
			return NULL; 
		}
	}
	return (PyObject *) self;
}


/* DESTRUCTOR: delete type object */
static void
OutputDescriptorObject_dealloc(OutputDescriptorObject *self)
{
	Py_XDECREF(self->dict);
	PyObject_Del(self);
}


/* Set attributes */
static int
OutputDescriptor_setattr(OutputDescriptorObject *self, char *name, PyObject *v)
{
	if (v == NULL) {
		int rv = PyDict_DelItemString(self->dict, name);
		if (rv < 0)
			PyErr_SetString(PyExc_AttributeError,"non-existing OutputDescriptor attribute");
		return rv;
	}
	else
		return PyDict_SetItemString(self->dict, name, v);
}


/* Get attributes */
static PyObject *
OutputDescriptor_getattr(OutputDescriptorObject *self, char *name)
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


/* String representation */
static PyObject *
OutputDescriptor_repr(PyObject *self)
{
	OutputDescriptorObject* v = (OutputDescriptorObject*)self;
	if (v->dict) return PyDict_Type.tp_repr((PyObject *)v->dict);
	else return PyString_FromString("OutputDescriptor()");
}

#define OutputDescriptor_alloc PyType_GenericAlloc
#define OutputDescriptor_free PyObject_Del


/*						REAL-TIME TYPE OBJECT						*/

PyTypeObject OutputDescriptor_Type = {
	PyObject_HEAD_INIT(NULL)
	0,						/*ob_size*/
	"vampy.OutputDescriptor",/*tp_name*/
	sizeof(OutputDescriptorObject),	/*tp_basicsize*/
	0,						/*tp_itemsize*/
	(destructor)OutputDescriptorObject_dealloc, /*tp_dealloc*/
	0,						/*tp_print*/
	(getattrfunc)OutputDescriptor_getattr, /*tp_getattr*/
	(setattrfunc)OutputDescriptor_setattr, /*tp_setattr*/
	0,						/*tp_compare*/
	OutputDescriptor_repr,	/*tp_repr*/
	0,						/*tp_as_number*/
	0,						/*tp_as_sequence*/
	0,						/*tp_as_mapping*/
	0,						/*tp_hash*/
	0,                      /*tp_call*/
    0,                      /*tp_str*/
    0,                      /*tp_getattro*/
    0,                      /*tp_setattro*/
    0,                      /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,     /*tp_flags*/
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
    0,                      /*tp_init*/
    OutputDescriptor_alloc, /*tp_alloc*/
    OutputDescriptor_new,   /*tp_new*/
    OutputDescriptor_free,	/*tp_free*/
    0,                      /*tp_is_gc*/
};

/*		  		 	  PyOutputDescriptor C++ API    				*/

