/* -*- c-basic-offset: 8 indent-tabs-mode: t -*- */
/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

/*
PyTypeInterface: Type safe conversion utilities between Python types 
and basic C/C++ types and Vamp API types.
*/

#ifndef _PY_TYPE_INTERFACE_H_
#define _PY_TYPE_INTERFACE_H_
#include <Python.h>
#ifdef HAVE_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#endif
#include "PyExtensionModule.h"
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include "vamp-sdk/Plugin.h"

using std::cerr;
using std::endl;

#ifdef HAVE_NUMPY
enum eArrayDataType {
	dtype_float32 = (int) NPY_FLOAT,
	dtype_complex64 = (int) NPY_CFLOAT 
	};
#endif 

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
	quantizeStep,
	valueNames
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
	timestamp,
	hasDuration,
	duration,
	values,
	label
	};

/* C++ mapping of PyNone Type */
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
		std::string str() const { 
			return (location.empty()) ? message : message + "\nLocation: " + location;}
		void print() const { cerr << str() << endl; }
		template<typename V> ValueError &operator<< (const V& v)
		{
			std::ostringstream ss;
			ss << v;
			location += ss.str();
			return *this;
		}
	};
	
	// Utilities
	void setStrictTypingFlag(bool b) {m_strict = b;}
	void setNumpyInstalled(bool b) {m_numpyInstalled = b;}
	ValueError getError() const;
	std::string PyValue_Get_TypeName(PyObject*) const;
	bool initMaps() const;

	// Basic type conversion: Python to C++ 
	float 	PyValue_To_Float(PyObject*) const;
	size_t 	PyValue_To_Size_t(PyObject*) const;
	bool 	PyValue_To_Bool(PyObject*) const;
	std::string PyValue_To_String(PyObject*) const;
	long 	PyValue_To_Long(PyObject*) const;
	// int 	PyValue_To_Int(PyObject* pyValue) const;
	
	
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

	// Input buffers to Python
	PyObject* InputBuffers_As_PythonLists(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype);
	PyObject* InputBuffers_As_SharedMemoryList(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype);

	// Numpy types
#ifdef HAVE_NUMPY
	std::vector<float> PyArray_To_FloatVector (PyObject *pyValue) const;
	PyObject* InputBuffers_As_NumpyArray(const float *const *inputBuffers, const size_t&, const size_t&, const Vamp::Plugin::InputDomain& dtype);
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
	
	//returns e.g. Vamp::Plugin::OutputDescriptor or Vamp::Plugin::Feature
	template<typename RET> 
	RET PyValue_To_VampDescriptor(PyObject* pyValue) const
	{
		PyObject* pyDict;

		// Descriptors encoded as dicts
		pyDict = GetDescriptor_As_Dict(pyValue);
		if (!pyDict) pyDict = pyValue;
	
		// TODO: support full mapping protocol as fallback.
		if (!PyDict_Check(pyDict)) {
			setValueError("Error while converting descriptor or feature object.\nThe value is neither a dictionary nor a Vamp Feature or Descriptor type.",m_strict);
#ifdef _DEBUG
			cerr << "PyTypeInterface::PyValue_To_VampDescriptor failed. Error: Unexpected return type." << endl;
#endif			
			return RET();
		}

		Py_ssize_t pyPos = 0;
		PyObject *pyKey, *pyDictValue;
		initMaps();
		int errors = 0;
		m_error = false;
		RET rd;

		//Python Dictionary Iterator:
		while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyDictValue))
		{
			std::string key = PyValue_To_String(pyKey);
#ifdef _DEBUG_VALUES			
			cerr << "key: '" << key << "' value: '" << PyValue_To_String(pyDictValue) << "' " << endl;
#endif			
			SetValue(rd,key,pyDictValue);
			if (m_error) {
				errors++;
				lastError() << "attribute '" << key << "'";// << " of " << getDescriptorId(rd);
			}
		}
		if (errors) {
			lastError() << " of " << getDescriptorId(rd);
			m_error = true;
#ifdef _DEBUG
			cerr << "PyTypeInterface::PyValue_To_VampDescriptor: Warning: Value error in descriptor." << endl;
#endif				
		}
		return rd;
	}

	/// Convert a sequence (tipically list) of PySomething to 
	/// OutputList,ParameterList or FeatureList
	/// <OutputList> <OutputDescriptor>
	template<typename RET,typename ELEM> 
	RET PyValue_To_VampList(PyObject* pyValue) const
	{
		RET list; // e.g. Vamp::Plugin::OutputList
		ELEM element; // e.g. Vamp::Plugin::OutputDescriptor

		/// convert lists (ParameterList, OutputList, FeatureList)
		if (PyList_Check(pyValue)) {
			PyObject *pyDict; //This reference will be borrowed
			m_error = false; int errors = 0;
			for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyValue); ++i) {
				//Get i-th Vamp output descriptor (Borrowed Reference)
				pyDict = PyList_GET_ITEM(pyValue,i);
				element = PyValue_To_VampDescriptor<ELEM>(pyDict);
				if (m_error) errors++;
				// Check for empty Feature/Descriptor as before?
				list.push_back(element);
			}
			if (errors) m_error=true;
			return list;
		}
		
		/// convert other types implementing the sequence protocol
		if (PySequence_Check(pyValue)) {
			PyObject *pySequence = PySequence_Fast(pyValue,"Returned value can not be converted to list or tuple.");
			PyObject **pyElements =  PySequence_Fast_ITEMS(pySequence);
			m_error = false; int errors = 0;
			for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(pySequence); ++i) 
			{
				element = PyValue_To_VampDescriptor<ELEM>(pyElements[i]);
				if (m_error) errors++;
				list.push_back(element);
			}
			if (errors) m_error=true;
			Py_XDECREF(pySequence);
			return list;
		}

		// accept None as an empty list
		if (pyValue == Py_None) return list;
		
		// in strict mode, returning a single value is not allowed 
		if (m_strict) {
			setValueError("Strict conversion error: object is not list or iterable sequence.",m_strict);
			return list;
		}
		
		/// try to insert single, non-iterable values. i.e. feature <- [feature]
		element = PyValue_To_VampDescriptor<ELEM>(pyValue);
		if (m_error) {
			setValueError("Could not insert returned value to Vamp List.",m_strict);
			return list; 
		}
		list.push_back(element);
		return list;
		
#ifdef _DEBUG
			cerr << "PyTypeInterface::PyValue_To_VampList failed. Expected iterable return type." << endl;
#endif			

	}

	/// Convert DTYPE type 1D NumpyArray to std::vector<RET>
	template<typename RET, typename DTYPE>
	std::vector<RET> PyArray_Convert(void* raw_data_ptr, long length, size_t strides) const
	{
		std::vector<RET> rValue;
		
		/// check if the array is continuous, if not use strides info
		if (sizeof(DTYPE)!=strides) {
#ifdef _DEBUG_VALUES
			cerr << "Warning: discontinuous numpy array. Strides: " << strides << " bytes. sizeof(dtype): " << sizeof(DTYPE) << endl;
#endif
			char* data = (char*) raw_data_ptr;
			for (long i = 0; i<length; ++i){
				rValue.push_back((RET)(*((DTYPE*)data)));
#ifdef _DEBUG_VALUES
				cerr << "value: " << (RET)(*((DTYPE*)data)) << endl;
#endif				
				data+=strides;
			}
			return rValue;
		}

		DTYPE* data = (DTYPE*) raw_data_ptr;
		for (long i = 0; i<length; ++i){
#ifdef _DEBUG_VALUES
			cerr << "value: " << (RET)data[i] << endl;
#endif
			rValue.push_back((RET)data[i]);
		}
		return rValue;
	}

#ifdef HAVE_NUMPY
	/// this is a special case. numpy.float64 has an array interface but no array descriptor
	inline std::vector<float> PyArray0D_Convert(PyArrayInterface *ai) const
	{
		std::vector<float> rValue;
		if ((ai->typekind) == *"f") 
			rValue.push_back((float)*(double*)(ai->data));
		else { 
			setValueError("Unsupported NumPy data type.",m_strict); 
			return rValue;
		}
#ifdef _DEBUG_VALUES
		cerr << "value: " << rValue[0] << endl;
#endif
		return rValue;
	}
#endif
	
	//Vamp specific types
	Vamp::Plugin::FeatureSet PyValue_To_FeatureSet(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::FeatureSet &r) const
		{ r = this->PyValue_To_FeatureSet(pyValue); }

	Vamp::RealTime PyValue_To_RealTime(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::RealTime &r) const
		{ r = this->PyValue_To_RealTime(pyValue); }
	
	Vamp::Plugin::OutputDescriptor::SampleType PyValue_To_SampleType(PyObject*) const;

	Vamp::Plugin::InputDomain PyValue_To_InputDomain(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::InputDomain &r) const
		{ r = this->PyValue_To_InputDomain(pyValue); }
	
	
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
	mutable bool m_error;
	mutable std::queue<ValueError> m_errorQueue;
	unsigned int m_inputSampleRate; 
	bool m_numpyInstalled;
	
	void setValueError(std::string,bool) const;
	ValueError& lastError() const;

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
	inline void _convert(PyObject *pyValue,Vamp::Plugin::OutputDescriptor::SampleType &r) const 
		{ r = PyValue_To_SampleType(pyValue); }
	// inline void _convert(PyObject *pyValue,Vamp::Plugin::InputDomain &r) const 
	// 	{ r = PyValue_To_InputDomain(pyValue); }
	    

	/* Identify descriptors for error reporting */
	inline std::string getDescriptorId(Vamp::Plugin::OutputDescriptor d) const
		{return std::string("Output Descriptor '") + d.identifier +"' ";}
	inline std::string getDescriptorId(Vamp::Plugin::ParameterDescriptor d) const
		{return std::string("Parameter Descriptor '") + d.identifier +"' ";}
	inline std::string getDescriptorId(Vamp::Plugin::Feature f) const
		{return std::string("Feature (") + f.label + ")"; }
	
public:
	const bool& error;

};

/* 		   		  Convert Sample Buffers to Python 	         		*/

/// passing the sample buffers as builtin python lists
/// Optimization: using fast sequence protocol
inline PyObject*
PyTypeInterface::InputBuffers_As_PythonLists(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype)
{
	//create a list of lists (new references)
	PyObject *pyChannelList = PyList_New((Py_ssize_t) channels);
	
	// Pack samples into a Python List Object
	// pyFloat/pyComplex types will always be new references, 
	// they will be freed when the lists are deallocated.
	
	PyObject **pyChannelListArray =  PySequence_Fast_ITEMS(pyChannelList);
	for (size_t i=0; i < channels; ++i) {
		
        size_t arraySize;
		if (dtype==Vamp::Plugin::FrequencyDomain) 
			arraySize = (blockSize / 2) + 1; //blockSize + 2; if cplx list isn't used
		else 
			arraySize = blockSize;

		PyObject *pySampleList = PyList_New((Py_ssize_t) arraySize);
		PyObject **pySampleListArray =  PySequence_Fast_ITEMS(pySampleList);
		
		// Note: passing a complex list crashes the C-style plugin
		// when it tries to convert it to a numpy array directly.
		// This plugin will be obsolete, but we have to find a way
		// to prevent such crash: possibly a numpy bug, 
		// works fine above 1.0.4
		
		switch (dtype) //(Vamp::Plugin::TimeDomain)
		{
			case Vamp::Plugin::TimeDomain :

			for (size_t j = 0; j < arraySize; ++j) {
				PyObject *pyFloat=PyFloat_FromDouble(
					(double) inputBuffers[i][j]);
				pySampleListArray[j] = pyFloat;
			}
			break;

			case Vamp::Plugin::FrequencyDomain :

			size_t k = 0;
			for (size_t j = 0; j < arraySize; ++j) {
				PyObject *pyComplex=PyComplex_FromDoubles(
					(double) inputBuffers[i][k], 
					(double) inputBuffers[i][k+1]);
				pySampleListArray[j] = pyComplex;
				k += 2;
			}
			break;
			
		}
		pyChannelListArray[i] = pySampleList;
	}
	return pyChannelList;
}

/// numpy buffer interface: passing the sample buffers as shared memory buffers
/// Optimization: using sequence protocol for creating the buffer list
inline PyObject*
PyTypeInterface::InputBuffers_As_SharedMemoryList(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype)
{	
	//create a list of buffers (returns new references)
	PyObject *pyChannelList = PyList_New((Py_ssize_t) channels);
	PyObject **pyChannelListArray =  PySequence_Fast_ITEMS(pyChannelList);

	// Expose memory using the Buffer Interface.		
	// This will pass a pointer which can be recasted in Python code 
	// as complex or float array using Numpy's frombuffer() method
	// (this will not copy values just keep the starting adresses 
	// for each channel in a list)
	Py_ssize_t bufferSize;
	
	if (dtype==Vamp::Plugin::FrequencyDomain) 
		bufferSize = (Py_ssize_t) sizeof(float) * (blockSize+2);
	else 
		bufferSize = (Py_ssize_t) sizeof(float) * blockSize;
	
	for (size_t i=0; i < channels; ++i) {
		PyObject *pyBuffer = PyBuffer_FromMemory
		((void *) (float *) inputBuffers[i],bufferSize);
		pyChannelListArray[i] = pyBuffer;
	}
	return pyChannelList;
}


/// numpy array interface: passing the sample buffers as 2D numpy array
/// Optimization: using array API (needs numpy headers)
#ifdef HAVE_NUMPY
inline PyObject*
PyTypeInterface::InputBuffers_As_NumpyArray(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype)
{	
/*
NOTE: We create a list of 1D Numpy arrays for each channel instead
of a matrix, because the address space of inputBuffers doesn't seem
to be continuous. Although the array strides could be calculated for
2 channels (i.e. inputBuffers[1] - inputBuffers[0]) i'm not sure
if this can be trusted, especially for more than 2 channels.

	cerr << "First channel: " << inputBuffers[0][0] << " address: " <<  inputBuffers[0] << endl;
	if (channels == 2)
		cerr << "Second channel: " << inputBuffers[1][0] << " address: " <<  inputBuffers[1] << endl;

*/	
	
	// create a list of arrays (returns new references)
	PyObject *pyChannelList = PyList_New((Py_ssize_t) channels);
	PyObject **pyChannelListArray =  PySequence_Fast_ITEMS(pyChannelList);
	
	// Expose memory using the Numpy Array Interface.		
	// This will wrap an array objects around the data.
	// (will not copy values just steal the starting adresses)

	int arraySize, typenum;
	
	switch (dtype)
	{
		case Vamp::Plugin::TimeDomain :
		typenum = dtype_float32; //NPY_FLOAT; 
		arraySize = (int) blockSize;
		break;

		case Vamp::Plugin::FrequencyDomain :
		typenum = dtype_complex64; //NPY_CFLOAT;
		arraySize = (int) (blockSize / 2) + 1;
		break;
		
		default :
		cerr << "PyTypeInterface::InputBuffers_As_NumpyArray: Error: Unsupported numpy array data type." << endl;
		return pyChannelList;
	}

	// size for each dimension
	npy_intp ndims[1]={arraySize}; 
	
	for (size_t i=0; i < channels; ++i) {
		PyObject *pyChannelArray = 
			//args: (dimensions, size in each dim, type kind, pointer to continuous array)
			PyArray_SimpleNewFromData(1, ndims, typenum, (void*) inputBuffers[i]);
		// make it read-only: set all flags to false except NPY_C_CONTIGUOUS
		//!!! what about NPY_ARRAY_OWNDATA?
		PyArray_CLEARFLAGS((PyArrayObject *)pyChannelArray, 0xff);
		PyArray_ENABLEFLAGS((PyArrayObject *)pyChannelArray, NPY_ARRAY_C_CONTIGUOUS);
		pyChannelListArray[i] = pyChannelArray;
	}
	return pyChannelList;
}
#endif



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
