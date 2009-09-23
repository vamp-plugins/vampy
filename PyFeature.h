#ifndef _PYFEATURE_H_
#define _PYFEATURE_H_

#include "vamp-sdk/Plugin.h"
// #include "PyTypeInterface.h"

typedef struct {
		PyObject_HEAD
		PyObject *dict; /* Attributes dictionary */
		// Vamp::Plugin::Feature *feature;
		/// pointer to type interface required
		// PyTypeInterface ti;
} FeatureObject; 

PyAPI_DATA(PyTypeObject) Feature_Type;

#define PyFeature_CheckExact(v)	((v)->ob_type == &Feature_Type)
#define PyFeature_Check(v) PyObject_TypeCheck(v, &Feature_Type)
///fast macro version as per API convention
#define PyFeature_AS_DICT(v) ((const FeatureObject* const) (v))->dict


/*		  		 	  PyFeature C++ API  	  		  				*/

/// Not required: we will never have to pass a feature back from the wrapper
// PyAPI_FUNC(PyObject *) 
// PyFeature_FromFeature(Vamp::RealTime&);

// PyAPI_FUNC(const Vamp::Plugin::Feature*) 
// PyFeature_AsFeature (PyObject *self);

///fast macro version as per API convention
// #define PyFeature_AS_FEATURE(v) ((const FeatureObject* const) (v))->feature


#endif
