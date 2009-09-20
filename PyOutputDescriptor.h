#ifndef _PYOUTPUTDESCRIPTOR_H_
#define _PYOUTPUTDESCRIPTOR_H_

#include "vamp-sdk/Plugin.h"

typedef struct {
		PyObject_HEAD
		PyObject *dict;
} OutputDescriptorObject; 

PyAPI_DATA(PyTypeObject) OutputDescriptor_Type;

#define PyOutputDescriptor_CheckExact(v)	((v)->ob_type == &OutputDescriptor_Type)
#define PyOutputDescriptor_Check(v) PyObject_TypeCheck(v, &OutputDescriptor_Type)

/*			  		 PyOutputDescriptor C++ API  	  	  	  		*/


///fast macro version as per API convention
#define PyOutputDescriptor_AS_DICT(v) ((const OutputDescriptorObject* const) (v))->dict

#endif