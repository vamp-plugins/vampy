/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#ifndef _PYPARAMETERDESCRIPTOR_H_
#define _PYPARAMETERDESCRIPTOR_H_

#include "vamp-sdk/Plugin.h"

typedef struct {
		PyObject_HEAD
		PyObject *dict;
} ParameterDescriptorObject; 

extern PyTypeObject ParameterDescriptor_Type;

#define PyParameterDescriptor_CheckExact(v)	((v)->ob_type == &ParameterDescriptor_Type)
#define PyParameterDescriptor_Check(v) PyObject_TypeCheck(v, &ParameterDescriptor_Type)

/*			  		 PyParameterDescriptor C++ API  	  		  	  */


///fast macro version as per API convention
#define PyParameterDescriptor_AS_DICT(v) ((const ParameterDescriptorObject* const) (v))->dict

#endif
