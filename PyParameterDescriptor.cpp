/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "PyParameterDescriptor.h"
#include "vamp-sdk/Plugin.h"
#include <string>
#include "PyTypeInterface.h"

using namespace std;
using namespace Vamp;
using Vamp::Plugin;

/*			 ParameterDescriptor Object's Methods 					*/ 
//these objects have no callable methods

/*		   PyParameterDescriptor methods implementing protocols   	*/ 
// these functions are called by the interpreter automatically

/* New ParameterDescriptor object */ 
static PyObject *
ParameterDescriptor_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{

	ParameterDescriptorObject *self = 
	(ParameterDescriptorObject*)type->tp_alloc(type, 0);
	
	if (self == NULL) return NULL;
    self->dict = PyDict_New();
	if (self->dict == NULL) return NULL;

	/// allow copying objects
    if (args && PyTuple_Size(args) == 1) {
		PyObject* arg = PyTuple_GET_ITEM(args,0);
		if (PyParameterDescriptor_CheckExact(arg))
			PyDict_Merge(self->dict,PyParameterDescriptor_AS_DICT(arg),0);
		else if (PyDict_CheckExact(arg)) 
			PyDict_Merge(self->dict,arg,0);
		else {
			PyErr_SetString(PyExc_TypeError, 
			"Object takes zero or one ParameterDescriptor or dictionary arguments.");
			return NULL; 
		}
	}
	return (PyObject *) self;
}


/* DESTRUCTOR: delete type object */
static void
ParameterDescriptorObject_dealloc(ParameterDescriptorObject *self)
{
	Py_XDECREF(self->dict);
	PyObject_Del(self);
}


/* Set attributes */
static int
ParameterDescriptor_setattr(ParameterDescriptorObject *self, char *name, PyObject *v)
{
	if (v == NULL) {
		int rv = PyDict_DelItemString(self->dict, name);
		if (rv < 0)
			PyErr_SetString(PyExc_AttributeError,"non-existing ParameterDescriptor attribute");
		return rv;
	}
	else
		return PyDict_SetItemString(self->dict, name, v);
}


/* Get attributes */
static PyObject *
ParameterDescriptor_getattr(ParameterDescriptorObject *self, char *name)
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
ParameterDescriptor_repr(PyObject *self)
{
	ParameterDescriptorObject* v = (ParameterDescriptorObject*)self;
	if (v->dict) return PyDict_Type.tp_repr((PyObject *)v->dict);
	else return PyString_FromString("ParameterDescriptor()");
}

#define ParameterDescriptor_alloc PyType_GenericAlloc
#define ParameterDescriptor_free PyObject_Del

PyTypeObject ParameterDescriptor_Type = {
	PyObject_HEAD_INIT(NULL)
	0,						/*ob_size*/
	"vampy.ParameterDescriptor",/*tp_name*/
	sizeof(ParameterDescriptorObject),	/*tp_basicsize*/
	0,						/*tp_itemsize*/
	(destructor)ParameterDescriptorObject_dealloc, /*tp_dealloc*/
	0,						/*tp_print*/
	(getattrfunc)ParameterDescriptor_getattr, /*tp_getattr*/
	(setattrfunc)ParameterDescriptor_setattr, /*tp_setattr*/
	0,						/*tp_compare*/
	ParameterDescriptor_repr,			/*tp_repr*/
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
    ParameterDescriptor_alloc,/*tp_alloc*/
    ParameterDescriptor_new,/*tp_new*/
    ParameterDescriptor_free,/*tp_free*/
    0,                      /*tp_is_gc*/
};

/*		  		 	  PyParameterDescriptor C++ API    				*/

