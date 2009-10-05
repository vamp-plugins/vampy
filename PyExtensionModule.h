/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#ifndef _PYEXTENSIONMODULE_H_
#define _PYEXTENSIONMODULE_H_

#include <Python.h>
#include <limits.h>
#include "PyRealTime.h"
#include "PyFeature.h"
#include "PyFeatureSet.h"
#include "PyParameterDescriptor.h"
#include "PyOutputDescriptor.h"

#ifndef UINT_MAX
#define UINT_MAX ((unsigned int) -1)
#endif
#define UINT_MAXD ((double) UINT_MAX)
/* long error() { std::cerr << "type error" << std::endl; return 0; } */
#define _dbl2uint(x) ((x) < 0 || (x) > UINT_MAXD ? 0 : (unsigned int)(x)+0.5)
#define _long2uint(x) ((x) < 0 || (x) > UINT_MAXD ? 0 : (unsigned int)(x))

using std::string;
using std::vector;

enum eVampyFlags {
	vf_NULL = 0,
	vf_DEBUG = 1, 
	vf_STRICT = 2,
	vf_QUIT = 4,
	vf_REALTIME = 8,
	vf_BUFFER = 16,
	vf_ARRAY = 32,
	vf_DEFAULT_V2 = (32 | 8)
};

#define PyDescriptor_Check(v) ((v)->ob_type == &Feature_Type) || ((v)->ob_type == &OutputDescriptor_Type) || ((v)->ob_type == &ParameterDescriptor_Type)

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initvampy();

#endif
