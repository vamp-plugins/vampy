/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#ifndef _PYFEATURESET_H_
#define _PYFEATURESET_H_

#include <Python.h>

typedef struct {
    PyDictObject dict;
} FeatureSetObject;

extern PyTypeObject FeatureSet_Type;

#define PyFeatureSet_CheckExact(v)	((v)->ob_type == &FeatureSet_Type)
#define PyFeatureSet_Check(v) PyObject_TypeCheck(v, &FeatureSet_Type)

extern void initFeatureSetType(void);

#endif
