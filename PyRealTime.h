/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#ifndef _PYREALTIME_H_
#define _PYREALTIME_H_

#include "vamp-sdk/Plugin.h"

typedef struct {
		PyObject_HEAD
		Vamp::RealTime *rt;
} RealTimeObject; 

PyAPI_DATA(PyTypeObject) RealTime_Type;

#define PyRealTime_CheckExact(v)	((v)->ob_type == &RealTime_Type)
#define PyRealTime_Check(v) PyObject_TypeCheck(v, &RealTime_Type)
///fast macro version as per API convention
#define PyRealTime_AS_REALTIME(v) ((const RealTimeObject* const) (v))->rt

/*		  		 	  PyRealTime C++ API  	  		  				*/


PyAPI_FUNC(PyObject *) 
PyRealTime_FromRealTime(const Vamp::RealTime&);

PyAPI_FUNC(const Vamp::RealTime*) 
PyRealTime_AsRealTime (PyObject *self);


#endif
