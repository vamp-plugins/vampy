#ifndef _PYREALTIME_H_
#define _PYREALTIME_H_

#include "vamp-sdk/Plugin.h"

/* RealTime Type Object's structure    */
/* Doc:: 10.2 Common Object Structures */
typedef struct {
		PyObject_HEAD
		/*PyObject	*rt_attrs;*/
		Vamp::RealTime::RealTime *rt;
} RealTimeObject; 

PyAPI_DATA(PyTypeObject) RealTime_Type;

#define PyRealTime_CheckExact(v)	((v)->ob_type == &RealTime_Type)
#define PyRealTime_Check(v) PyObject_TypeCheck(v, &RealTime_Type)

/* pyRealTime C API functions */
//	Example from Python's stringobject.h
// 	PyAPI_FUNC(PyObject *) PyString_FromString(const char *);

#ifdef __cplusplus
extern "C" {
#endif
	
PyAPI_FUNC(PyObject *) 
PyRealTime_FromRealTime(Vamp::RealTime *rt);

PyAPI_FUNC(Vamp::RealTime::RealTime *) 
PyRealTime_AsPointer (PyObject *self);

/* PyRealTime Module functions */

PyAPI_FUNC(PyObject *)
RealTime_new(PyObject *ignored, PyObject *args);

PyAPI_FUNC(PyObject *)
RealTime_frame2RealTime(PyObject *ignored, PyObject *args);

#ifdef __cplusplus
}
#endif
#endif /* _PYREALTIME_H_ */