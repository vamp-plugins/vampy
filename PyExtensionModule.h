#ifndef _PYEXTENSIONMODULE_H_
#define _PYEXTENSIONMODULE_H_

#include <Python.h>
#include "PyRealTime.h"
#include "PyFeature.h"
#include "PyFeatureSet.h"
#include "PyParameterDescriptor.h"
#include "PyOutputDescriptor.h"

#define PyDescriptor_Check(v) ((v)->ob_type == &Feature_Type) || ((v)->ob_type == &OutputDescriptor_Type) || ((v)->ob_type == &ParameterDescriptor_Type)

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initvampy();
void cleanModule();

#endif
