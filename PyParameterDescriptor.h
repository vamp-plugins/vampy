#ifndef _PYPARAMETERDESCRIPTOR_H_
#define _PYPARAMETERDESCRIPTOR_H_

#include "vamp-sdk/Plugin.h"

typedef struct {
		PyObject_HEAD
		PyObject *dict;
} ParameterDescriptorObject; 

PyAPI_DATA(PyTypeObject) ParameterDescriptor_Type;

#define PyParameterDescriptor_CheckExact(v)	((v)->ob_type == &ParameterDescriptor_Type)
#define PyParameterDescriptor_Check(v) PyObject_TypeCheck(v, &ParameterDescriptor_Type)

/*			  		 PyParameterDescriptor C++ API  	  		  	  */


///fast macro version as per API convention
#define PyParameterDescriptor_AS_DICT(v) ((const ParameterDescriptorObject* const) (v))->dict

#endif