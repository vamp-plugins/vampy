/*

Type safe conversion utilities from Python types to C/C++ types,
mainly using Py/C API macros.

*/

#ifndef _PY_TYPE_INTERFACE_H_
#define _PY_TYPE_INTERFACE_H_
#include <Python.h>
#ifdef HAVE_NUMPY
#include "arrayobject.h"
#endif
#include "PyExtensionModule.h"
#include <vector>
#include <queue>
#include <string>
#include "vamp-sdk/Plugin.h"
//#include <typeinfo>


using std::cerr;
using std::endl;

namespace o {
enum eOutDescriptors {
	not_found,
	identifier,
	name,
	description,
	unit, 
	hasFixedBinCount,
	binCount,
	binNames,
	hasKnownExtents,
	minValue,
	maxValue,
	isQuantized,
	quantizeStep,
	sampleType,	
	sampleRate,
	hasDuration,
	endNode
	}; 
}

namespace p {
enum eParmDescriptors {
	not_found,
	identifier,
	name,
	description,
	unit, 
	minValue,
	maxValue,
	defaultValue,
	isQuantized,
	quantizeStep
	};
}

enum eSampleTypes {
	OneSamplePerStep,
	FixedSampleRate,
	VariableSampleRate
	};

enum eFeatureFields {
	unknown,
	hasTimestamp,
	timeStamp,
	hasDuration,
	duration,
	values,
	label
	};

/* C++ mapping of PyNone Type*/
struct NoneType {};

class PyTypeInterface
{
public:
	PyTypeInterface();
	~PyTypeInterface();
	
	// Data
	class ValueError
	{
	public:
		ValueError() {}
		ValueError(std::string m, bool s) : message(m),strict(s) {}
		std::string location;
		std::string message;
		bool strict;
		std::string get() const { return message + "\nLocation: " + location + "\n";}
		void print() const { cerr << get(); }
	};
	
	// Utilities
	void setStrictTypingFlag(bool b) {m_strict = b;}
	const ValueError &lastError() const;
	ValueError getError() const;
	std::string PyValue_Get_TypeName(PyObject*) const;
	bool initMaps() const;

	// Basic type conversion: Python to C++ 
	float 	PyValue_To_Float(PyObject*) const;
	size_t 	PyValue_To_Size_t(PyObject*) const;
	bool 	PyValue_To_Bool(PyObject*) const;
	std::string PyValue_To_String(PyObject*) const;
	// int PyValue_To_Int(PyObject*) const;
	
	// C++ to Python
	PyObject *PyValue_From_CValue(const char*) const;
	PyObject *PyValue_From_CValue(const std::string& x) const { return PyValue_From_CValue(x.c_str()); }
	PyObject *PyValue_From_CValue(size_t) const;
	PyObject *PyValue_From_CValue(double) const;
	PyObject *PyValue_From_CValue(float x) const { return PyValue_From_CValue((double)x); }
	PyObject *PyValue_From_CValue(bool) const;
	
	// Sequence types
	std::vector<std::string> PyValue_To_StringVector (PyObject*) const;
	std::vector<float> PyValue_To_FloatVector (PyObject*) const;
	std::vector<float> PyList_To_FloatVector (PyObject*) const;

	// Numpy types
#ifdef HAVE_NUMPY
	std::vector<float> PyArray_To_FloatVector (PyObject *pyValue) const;
#endif	
	

/* 						Template functions 							*/


	/// Common wrappers to set values in Vamp API structs. (to be used in template functions)
	void SetValue(Vamp::Plugin::OutputDescriptor& od, std::string& key, PyObject* pyValue) const;
	void SetValue(Vamp::Plugin::ParameterDescriptor& od, std::string& key, PyObject* pyValue) const;
	bool SetValue(Vamp::Plugin::Feature& od, std::string& key, PyObject* pyValue) const;
    PyObject* GetDescriptor_As_Dict(PyObject* pyValue) const 
	{
		if PyFeature_CheckExact(pyValue) return PyFeature_AS_DICT(pyValue);
		if PyOutputDescriptor_CheckExact(pyValue) return PyOutputDescriptor_AS_DICT(pyValue);
		if PyParameterDescriptor_CheckExact(pyValue) return PyParameterDescriptor_AS_DICT(pyValue);
		return NULL;
	}
	

	template<typename RET> 
	RET PyValue_To_VampDescriptor(PyObject* pyValue) const
	//returns e.g. Vamp::Plugin::OutputDescriptor or Vamp::Plugin::Feature
	{
		PyObject* pyDict;

		// Descriptors encoded as dicts
		pyDict = GetDescriptor_As_Dict(pyValue);
		if (!pyDict) pyDict = pyValue;
	
		// TODO: support full mapping protocol as fallback.
		if (!PyDict_Check(pyDict)) {
			setValueError("Error while converting descriptor or feature object.\nThe value is neither a dictionary nor a Vamp Feature or Descriptor type.",m_strict);
			return RET();
		}

		Py_ssize_t pyPos = 0;
		PyObject *pyKey, *pyDictValue;
		initMaps();
		RET rd;

		//Python Dictionary Iterator:
		while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyDictValue))
		{
			std::string key = PyValue_To_String(pyKey);
			SetValue(rd,key,pyDictValue);
			if (m_error) {
				_lastError().location += "parameter: '" + key + "'";//"' descriptor: '" + rd.identifier + "'";
			}
		}
		if (!m_errorQueue.empty()) m_error = true;
		return rd;
	}

	/// Convert a sequence (tipically list) of PySomething to 
	/// OutputList,ParameterList or FeatureList
	template<typename RET,typename ELEM> //<OutputList> <OutputDescriptor>

	RET PyValue_To_VampList(PyObject* pyList) const
	{
		// Vamp::Plugin::OutputList list;
		// Vamp::Plugin::OutputDescriptor od;
		RET list;
		ELEM element;

		// Type checking
		if (! PyList_Check(pyList) ) {
			Py_CLEAR(pyList);
			// cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			// << "] Expected List return type." << endl;
			return list;
		}

		//This reference will be borrowed
		PyObject *pyDict;

		//Parse Output List
		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
			//Get i-th Vamp output descriptor (Borrowed Reference)
			pyDict = PyList_GET_ITEM(pyList,i);
			element = PyValue_To_VampDescriptor<ELEM>(pyDict);
			// Check for empty Feature/Descriptor as before?
			list.push_back(element);
		}
		return list;
	}

	/// Convert DTYPE type 1D NumpyArray to std::vector<RET>
	template<typename RET, typename DTYPE>
	std::vector<RET> PyArray_Convert(char* raw_data_ptr, long length) const
	{
		std::vector<RET> rValue;
		DTYPE* data = (DTYPE*) raw_data_ptr;
		for (long i = 0; i<length; ++i){
#ifdef _DEBUG
			cerr << "value: " << (RET)data[i] << endl;
#endif
			rValue.push_back((RET)data[i]);
		}
		return rValue;
	}
	
	//Vamp specific types

	Vamp::Plugin::FeatureSet PyValue_To_FeatureSet(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::FeatureSet &r) const
		{ r = this->PyValue_To_FeatureSet(pyValue); }

	Vamp::RealTime PyValue_To_RealTime(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::RealTime &r) const
		{ r = this->PyValue_To_RealTime(pyValue); }
	
	
	/* Overloaded PyValue_To_rValue() to support generic functions */
	inline void PyValue_To_rValue(PyObject *pyValue, float &defValue) const 
		{ float tmp = this->PyValue_To_Float(pyValue);                                              
			if(!m_error) defValue = tmp; }
	inline void PyValue_To_rValue(PyObject *pyValue, size_t &defValue) const
		{ size_t tmp = this->PyValue_To_Size_t(pyValue); 
			if(!m_error) defValue = tmp; }
	inline void PyValue_To_rValue(PyObject *pyValue, bool &defValue) const
		{ bool tmp = this->PyValue_To_Bool(pyValue); 
			if(!m_error) defValue = tmp; }
	inline void PyValue_To_rValue(PyObject *pyValue, std::string &defValue) const
		{ std::string tmp = this->PyValue_To_String(pyValue); 
			if(!m_error) defValue = tmp; }
	/*used by templates where we expect no return value, if there is one it will be ignored*/			
	inline void PyValue_To_rValue(PyObject *pyValue, NoneType &defValue) const
		{ if (m_strict && pyValue != Py_None) 
				setValueError("Strict conversion error: Expected 'None' type.",m_strict); 
		}

	/* convert sequence types to Vamp List types */			
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::OutputList &r) const
		{ r = this->PyValue_To_VampList<Vamp::Plugin::OutputList,Vamp::Plugin::OutputDescriptor>(pyValue); }
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::ParameterList &r) const
		{ r = this->PyValue_To_VampList<Vamp::Plugin::ParameterList,Vamp::Plugin::ParameterDescriptor>(pyValue); }
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::FeatureList &r) const
		{ r = this->PyValue_To_VampList<Vamp::Plugin::FeatureList,Vamp::Plugin::Feature>(pyValue); }
	
	/// this is only needed for RealTime->Frame conversion
	void setInputSampleRate(float inputSampleRate)
		{ m_inputSampleRate = (unsigned int) inputSampleRate; }
	
private:
	bool m_strict;
	ValueError m_noError;
	mutable bool m_error;
	mutable ValueError& m_lastError;
	mutable std::queue<ValueError> m_errorQueue;
	// we only use it for RealTime conversion which requires unsigned int
	unsigned int m_inputSampleRate; 
	
	void setValueError(std::string,bool) const;
	ValueError& _lastError() const;

	/* Overloaded _convert(), bypasses error checking to avoid doing it twice in internals. */
	inline void _convert(PyObject *pyValue,float &r) const 
		{ r = PyValue_To_Float(pyValue); }
	inline void _convert(PyObject *pyValue,size_t &r) const 
		{ r = PyValue_To_Size_t(pyValue); }
    inline void _convert(PyObject *pyValue,bool &r) const 
		{ r = PyValue_To_Bool(pyValue); }
	inline void _convert(PyObject *pyValue,std::string &r) const
		{ r = PyValue_To_String(pyValue); }
	inline void _convert(PyObject *pyValue,std::vector<std::string> &r) const
		{ r = PyValue_To_StringVector(pyValue); }
	inline void _convert(PyObject *pyValue,std::vector<float> &r) const
		{ r = PyValue_To_FloatVector(pyValue); }
    inline void _convert(PyObject *pyValue,Vamp::RealTime &r) const 
		{ r = PyValue_To_RealTime(pyValue); }
	
public:
	const bool& error;

};

#ifdef NUMPY_REFERENCE
/// This should be all we need to compile without direct dependency,
/// but we don't do that. (it may not work on some platforms)
typedef struct {
    int two;              /* contains the integer 2 -- simple sanity check */
    int nd;               /* number of dimensions */
    char typekind;        /* kind in array --- character code of typestr */
    int itemsize;         /* size of each element */
    int flags;            /* flags indicating how the data should be interpreted */
                          /*   must set ARR_HAS_DESCR bit to validate descr */
    Py_intptr_t *shape;   /* A length-nd array of shape information */
    Py_intptr_t *strides; /* A length-nd array of stride information */
    void *data;           /* A pointer to the first element of the array */
    PyObject *descr;      /* NULL or data-description (same as descr key */
                          /*        of __array_interface__) -- must set ARR_HAS_DESCR */
                          /*        flag or this will be ignored. */
} PyArrayInterface;

typedef struct PyArrayObject {
        PyObject_HEAD
        char *data;             /* pointer to raw data buffer */
        int nd;                 /* number of dimensions, also called ndim */
        npy_intp *dimensions;       /* size in each dimension */
        npy_intp *strides;          /* bytes to jump to get to the
                                   next element in each dimension */
        PyObject *base;         /* This object should be decref'd
                                   upon deletion of array */
                                /* For views it points to the original array */
                                /* For creation from buffer object it points
                                   to an object that shold be decref'd on
                                   deletion */
                                /* For UPDATEIFCOPY flag this is an array
                                   to-be-updated upon deletion of this one */
        PyArray_Descr *descr;   /* Pointer to type structure */
        int flags;              /* Flags describing array -- see below*/
        PyObject *weakreflist;  /* For weakreferences */
} PyArrayObject;

typedef struct _PyArray_Descr {
        PyObject_HEAD
        PyTypeObject *typeobj;  /* the type object representing an
                                   instance of this type -- should not
                                   be two type_numbers with the same type
                                   object. */
        char kind;              /* kind for this type */
        char type;              /* unique-character representing this type */
        char byteorder;         /* '>' (big), '<' (little), '|'
                                   (not-applicable), or '=' (native). */
        char hasobject;         /* non-zero if it has object arrays
                                   in fields */
        int type_num;          /* number representing this type */
        int elsize;             /* element size for this type */
        int alignment;          /* alignment needed for this type */
        struct _arr_descr                                       \
        *subarray;              /* Non-NULL if this type is
                                   is an array (C-contiguous)
                                   of some other type
                                */
        PyObject *fields;       /* The fields dictionary for this type */
                                /* For statically defined descr this
                                   is always Py_None */

        PyObject *names;        /* An ordered tuple of field names or NULL
                                   if no fields are defined */

        PyArray_ArrFuncs *f;     /* a table of functions specific for each
                                    basic data descriptor */
} PyArray_Descr;

enum NPY_TYPES {    NPY_BOOL=0,
                    NPY_BYTE, NPY_UBYTE,
                    NPY_SHORT, NPY_USHORT,
                    NPY_INT, NPY_UINT,
                    NPY_LONG, NPY_ULONG,
                    NPY_LONGLONG, NPY_ULONGLONG,
                    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                    NPY_OBJECT=17,
                    NPY_STRING, NPY_UNICODE,
                    NPY_VOID,
                    NPY_NTYPES,
                    NPY_NOTYPE,
                    NPY_CHAR,      /* special flag */
                    NPY_USERDEF=256  /* leave room for characters */
};
#endif /*NUMPY_REFERENCE*/
#endif
