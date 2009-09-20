#ifndef _PYFEATURESET_H_
#define _PYFEATURESET_H_

#include <Python.h>

typedef struct {
    PyDictObject dict;
    int state;
} FeatureSetObject;

PyAPI_DATA(PyTypeObject) FeatureSet_Type;

#define PyFeatureSet_CheckExact(v)	((v)->ob_type == &FeatureSet_Type)
#define PyFeatureSet_Check(v) PyObject_TypeCheck(v, &FeatureSet_Type)
// #define PyFeatureSet_CheckExact(v)	((v)->ob_type == &PyDict_Type)
// #define PyFeatureSet_Check(v) PyObject_TypeCheck(v, &PyDict_Type)

// #define PyFeature_AS_DICT(v) ((const FeatureObject* const) (v))->dict

void initFeatureSetType(void);

#endif