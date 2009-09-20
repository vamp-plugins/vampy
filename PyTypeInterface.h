/*

Type safe conversion utilities from Python types to C/C++ types,
mainly using Py/C API macros.

*/

#ifndef _PY_TYPE_INTERFACE_H_
#define _PY_TYPE_INTERFACE_H_

#include "vamp-sdk/Plugin.h"
#include <Python.h>
#include "PyExtensionModule.h"
#include <vector>
#include <queue>
#include <string>
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


/// sutructure of NumPy array interface:
/// this is all we need to support numpy without direct dependency
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

/* C++ mapping of PyNone Type*/
typedef struct NoneType {};

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

	// Numpy types
	float* getNumPyObjectData(PyObject *object, int &length) const; 
	

/* 						Template functions 							*/


	/// Common wrappers to set a value in one of these structs. (to be used in template functions)
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
	RET PyTypeInterface::PyValue_To_VampDescriptor(PyObject* pyValue) const
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
	RET PyTypeInterface::PyValue_To_VampList(PyObject* pyList) const
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

	
	//Vamp specific types

	Vamp::Plugin::FeatureSet PyValue_To_FeatureSet(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::Plugin::FeatureSet &r) const
		{ r = this->PyValue_To_FeatureSet(pyValue); }

	Vamp::RealTime::RealTime PyValue_To_RealTime(PyObject*) const;
	inline void PyValue_To_rValue(PyObject *pyValue, Vamp::RealTime::RealTime &r) const
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
				setValueError("Strict conversion error: expected 'None' type.",m_strict); 
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
    inline void _convert(PyObject *pyValue,Vamp::RealTime::RealTime &r) const 
		{ r = PyValue_To_RealTime(pyValue); }
	
public:
	const bool& error;

};

#endif