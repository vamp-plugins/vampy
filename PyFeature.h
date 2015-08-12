/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

/*
NOTES: There are two ways to implement the Vamp::Feature wrapper.
1) We could keep a Vamp::Feature in the object and
convert the values on the fly as they are inserted.
However, this requires a way to convert back to Python for
this object to be fully usable in python code. These conversions 
are otherwise unnecessary.

2) Keep the python attribute objects in a dict as it is normally 
done in python classes, and convert when the object is returned.
This way the object is usable by the interpreter until it is returned
to the C++ plugin wrapper.
This is different form the Vampy:PyRealTime implementation where the
two-way conversion makes more sense (in fact required). Note: For
a host implementation option 1) will be required.

*/

#ifndef _PYFEATURE_H_
#define _PYFEATURE_H_

#include "vamp-sdk/Plugin.h"
// #include "PyTypeInterface.h"


typedef struct {
		PyObject_HEAD
		PyObject *dict; 
		// Vamp::Plugin::Feature *feature;
		/// pointer to type interface required: PyTypeInterface ti;
} FeatureObject; 

extern PyTypeObject Feature_Type;

#define PyFeature_CheckExact(v)	((v)->ob_type == &Feature_Type)
#define PyFeature_Check(v) PyObject_TypeCheck(v, &Feature_Type)

///fast macro version as per API convention
#define PyFeature_AS_DICT(v) ((const FeatureObject* const) (v))->dict
// #define PyFeature_AS_FEATURE(v) ((const FeatureObject* const) (v))->feature
 

/*		  		 	  PyFeature C++ API  	  		  				*/

/* Not required here: 
 	we will never have to pass a feature back from the wrapper */
// PyAPI_FUNC(PyObject *) 
// PyFeature_FromFeature(Vamp::Plugin::Feature&);

// PyAPI_FUNC(const Vamp::Plugin::Feature*) 
// PyFeature_AsFeature (PyObject *self);



#endif
