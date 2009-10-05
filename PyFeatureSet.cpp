/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "PyFeatureSet.h"
#include "vamp-sdk/Plugin.h"

using namespace std;

static int
FeatureSet_init(FeatureSetObject *self, PyObject *args, PyObject *kwds)
{
    if (PyDict_Type.tp_init((PyObject *)self, args, kwds) < 0)
        return -1;
    return 0;
}

static int
FeatureSetObject_ass_sub(FeatureSetObject *mp, PyObject *v, PyObject *w)
{
	if (!PyInt_CheckExact(v)) {
		PyErr_SetString(PyExc_ValueError,
			"Output index must be positive integer.");
		return 0;
	}
	if (w == NULL)
		return PyDict_DelItem((PyObject *)mp, v);
	else
		return PyDict_SetItem((PyObject *)mp, v, w);
}

#define FeatureSet_alloc PyType_GenericAlloc
#define FeatureSet_free PyObject_Del
//#define FeatureSet_as_mapping PyDict_Type.tp_as_mapping

static PyMappingMethods FeatureSet_as_mapping = *(PyDict_Type.tp_as_mapping);

PyTypeObject FeatureSet_Type = PyDict_Type;

void
initFeatureSetType(void)
{
	/*This type is derived from PyDict. We just override some slots here.*/
	/*The typical use case is index based assignment as opposed to object memeber access.*/
	FeatureSet_Type.ob_type = &PyType_Type;
	FeatureSet_Type.tp_base = &PyDict_Type;
	FeatureSet_Type.tp_bases = PyTuple_Pack(1, FeatureSet_Type.tp_base);
	FeatureSet_Type.tp_name = "vampy.FeatureSet";
	// FeatureSet_Type.tp_new = FeatureSet_new;
	FeatureSet_Type.tp_init = (initproc)FeatureSet_init;
	FeatureSet_Type.tp_basicsize = sizeof(FeatureSetObject);
	FeatureSet_as_mapping.mp_ass_subscript = (objobjargproc)FeatureSetObject_ass_sub;
	FeatureSet_Type.tp_as_mapping = &FeatureSet_as_mapping;
}

