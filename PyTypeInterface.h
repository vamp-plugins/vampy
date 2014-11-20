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
and Vamp API types. See PyTypeConversions for basic C/C++ types.
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
#include "PyTypeConversions.h"
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include "vamp-sdk/Plugin.h"

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

class PyTypeInterface
{
	PyTypeConversions m_conv;
	
public:
	PyTypeInterface();
	~PyTypeInterface();
	
	// Utilities
	void setStrictTypingFlag(bool b) {m_strict = b; m_conv.setStrictTypingFlag(b);}
	void setNumpyInstalled(bool b) {m_numpyInstalled = b; m_conv.setNumpyInstalled(b); }
	ValueError getError() const;
	std::string PyValue_Get_TypeName(PyObject*) const;
	bool initMaps() const;

	// Input buffers to Python
	PyObject* InputBuffers_As_PythonLists(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype);
	PyObject* InputBuffers_As_SharedMemoryList(const float *const *inputBuffers,const size_t& channels, const size_t& blockSize, const Vamp::Plugin::InputDomain& dtype);

	// Numpy types
#ifdef HAVE_NUMPY
	PyObject* InputBuffers_As_NumpyArray(const float *const *inputBuffers, const size_t&, const size_t&, const Vamp::Plugin::InputDomain& dtype);
#endif

/* Template functions */


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
			std::string key = m_conv.PyValue_To_String(pyKey);
#ifdef _DEBUG_VALUES			
			cerr << "key: '" << key << "' value: '" << m_conv.PyValue_To_String(pyDictValue) << "' " << endl;
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
		{ float tmp = m_conv.PyValue_To_Float(pyValue);                                              
			if(!m_error) defValue = tmp; }
	inline void PyValue_To_rValue(PyObject *pyValue, size_t &defValue) const
		{ size_t tmp = m_conv.PyValue_To_Size_t(pyValue); 
			if(!m_error) defValue = tmp; }
	inline void PyValue_To_rValue(PyObject *pyValue, bool &defValue) const
		{ bool tmp = m_conv.PyValue_To_Bool(pyValue); 
			if(!m_error) defValue = tmp; }
	inline void PyValue_To_rValue(PyObject *pyValue, std::string &defValue) const
		{ std::string tmp = m_conv.PyValue_To_String(pyValue); 
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
		{ r = m_conv.PyValue_To_Float(pyValue); }
	inline void _convert(PyObject *pyValue,size_t &r) const 
		{ r = m_conv.PyValue_To_Size_t(pyValue); }
	inline void _convert(PyObject *pyValue,bool &r) const 
		{ r = m_conv.PyValue_To_Bool(pyValue); }
	inline void _convert(PyObject *pyValue,std::string &r) const
		{ r = m_conv.PyValue_To_String(pyValue); }
	inline void _convert(PyObject *pyValue,std::vector<std::string> &r) const
		{ r = m_conv.PyValue_To_StringVector(pyValue); }
	inline void _convert(PyObject *pyValue,std::vector<float> &r) const
		{ r = m_conv.PyValue_To_FloatVector(pyValue); }
	inline void _convert(PyObject *pyValue,Vamp::RealTime &r) const 
		{ r = PyValue_To_RealTime(pyValue); }
	inline void _convert(PyObject *pyValue,Vamp::Plugin::OutputDescriptor::SampleType &r) const 
		{ r = PyValue_To_SampleType(pyValue); }
	// inline void _convert(PyObject *pyValue,Vamp::Plugin::InputDomain &r) const 
	// 	{ r = m_conv.PyValue_To_InputDomain(pyValue); }
	    

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

#endif
