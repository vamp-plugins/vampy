#include <Python.h>
#include "PyFeatureSet.h"
#include "vamp-sdk/Plugin.h"

using namespace std;

static int
FeatureSet_init(FeatureSetObject *self, PyObject *args, PyObject *kwds)
{
    if (PyDict_Type.tp_init((PyObject *)self, args, kwds) < 0)
        return -1;
    self->state = 0;
	cerr << "FeatureSet initialised" << endl;
    return 0;
}

static int
FeatureSetObject_ass_sub(FeatureSetObject *mp, PyObject *v, PyObject *w)
{
	// cerr << "called FeatureSetObject_ass_sub" << endl;
	if (!PyInt_CheckExact(v)) {
		/// TODO: Set ValueError here.
		cerr << "Output index must be positive integer" << endl;
		return 0;
	}
	if (w == NULL)
		return PyDict_DelItem((PyObject *)mp, v);
	else
		return PyDict_SetItem((PyObject *)mp, v, w);
}

#define FeatureSet_alloc PyType_GenericAlloc
#define FeatureSet_free PyObject_Del
//#define FeatureSet_as_mapping PyDict_Type.tp_as_mapping

static PyMappingMethods FeatureSet_as_mapping = *(PyDict_Type.tp_as_mapping);

PyTypeObject FeatureSet_Type = PyDict_Type;
// PyTypeObject FeatureSet_Type = {
// 	PyObject_HEAD_INIT(NULL)
// 	0,						/*ob_size*/
// 	"vampy.FeatureSet",		/*tp_name*/
// 	sizeof(FeatureSetObject),	/*tp_basicsize*/
// 	0,						/*tp_itemsize*/
// 	(destructor)FeatureSetObject_dealloc, /*tp_dealloc*/
// 	0,//PyDict_Type.tp_print,	/*tp_print*/
// 	0,//PyDict_Type.tp_getattr, /*tp_getattr*/
// 	0,//PyDict_Type.tp_setattr, /*tp_setattr*/
// 	0,						/*tp_compare*/
// 	0,//PyDict_Type.tp_repr,	/*tp_repr*/
// 	0,						/*tp_as_number*/
// 	0,						/*tp_as_sequence*/
// 	FeatureSet_as_mapping,	/*tp_as_mapping*/
// 	0,						/*tp_hash*/
// 	0,//Feature_test,           /*tp_call*/ // call on an instance
//     0,                      /*tp_str*/
//     PyDict_Type.tp_getattro,/*tp_getattro*/
//     0,//PyDict_Type.tp_setattro,/*tp_setattro*/
//     0,                      /*tp_as_buffer*/
//     Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /*tp_flags*/
//     0,                      /*tp_doc*/
//     PyDict_Type.tp_traverse,                      /*tp_traverse*/
//     PyDict_Type.tp_clear,                      /*tp_clear*/
//     0,                      /*tp_richcompare*/
//     0,                      /*tp_weaklistoffset*/
//     0,                      /*tp_iter*/
//     0,                      /*tp_iternext*/
//     PyDict_Type.tp_methods,	/*tp_methods*/ //TypeObject Methods
//     PyDict_Type.tp_members, /*tp_members*/
//     PyDict_Type.tp_getset,  /*tp_getset*/
//     0,                      /*tp_base*/
//     PyDict_Type.tp_dict,    /*tp_dict*/
//     0,                      /*tp_descr_get*/
//     0,                      /*tp_descr_set*/
//     PyDict_Type.tp_dictoffset, /*tp_dictoffset*/
//     (initproc)FeatureSet_init, /*tp_init*/
//     FeatureSet_alloc,          /*tp_alloc*/
//     FeatureSet_new,            /*tp_new*/
//     FeatureSet_free,			/*tp_free*/
//     0,                      /*tp_is_gc*/
// };


void
initFeatureSetType(void)
{
	/*This type is derived from PyDict. We just override some slots here.*/
	/*The typical use case is index based assignment as opposed to object memeber access.*/
	FeatureSet_Type.ob_type = &PyType_Type;
	FeatureSet_Type.tp_base = &PyDict_Type;
	FeatureSet_Type.tp_bases = PyTuple_Pack(1, FeatureSet_Type.tp_base);
	FeatureSet_Type.tp_name = "vampy.FeatureSet";
	// FeatureSet_Type.tp_new = FeatureSet_new;
	FeatureSet_Type.tp_init = (initproc)FeatureSet_init;
	FeatureSet_Type.tp_basicsize = sizeof(FeatureSetObject);
	FeatureSet_as_mapping.mp_ass_subscript = (objobjargproc)FeatureSetObject_ass_sub;
	FeatureSet_Type.tp_as_mapping = &FeatureSet_as_mapping;
}

