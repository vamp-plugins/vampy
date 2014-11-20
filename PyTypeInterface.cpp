/* -*- c-basic-offset: 8 indent-tabs-mode: t -*- */
/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>

#ifdef HAVE_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#endif

#include "PyTypeInterface.h"
#include "PyRealTime.h"
#include "PyExtensionModule.h"
#include <math.h>
#include <float.h>
#include <limits.h>
#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::map;

static std::map<std::string, o::eOutDescriptors> outKeys;
static std::map<std::string, p::eParmDescriptors> parmKeys;
static std::map<std::string, eSampleTypes> sampleKeys;
static std::map<std::string, eFeatureFields> ffKeys;
static bool isMapInitialised = false;

/*  Note: NO FUNCTION IN THIS CLASS SHOULD ALTER REFERENCE COUNTS
	(EXCEPT FOR TEMPORARY PYTHON OBJECTS)! */

PyTypeInterface::PyTypeInterface() : 
	m_strict(false),
	m_error(false),
	m_numpyInstalled(false),
	error(m_error) // const public reference for easy access
{
}

PyTypeInterface::~PyTypeInterface()
{
}

/// FeatureSet (an integer map of FeatureLists)
Vamp::Plugin::FeatureSet
PyTypeInterface::PyValue_To_FeatureSet(PyObject* pyValue) const
{
	Vamp::Plugin::FeatureSet rFeatureSet;

	/// Convert PyFeatureSet 
	if (PyFeatureSet_CheckExact(pyValue)) {

		Py_ssize_t pyPos = 0;
		PyObject *pyKey, *pyDictValue; // Borrowed References
		int key;
		// bool it_error = false;

		m_error = false;
		while (PyDict_Next(pyValue, &pyPos, &pyKey, &pyDictValue))
		{
			key = (int) PyInt_AS_LONG(pyKey);
#ifdef _DEBUG_VALUES			
			cerr << "key: '" << key << "' value: '" << PyValue_To_String(pyDictValue) << "' " << endl;
#endif			
			// DictValue -> Vamp::FeatureList
			PyValue_To_rValue(pyDictValue,rFeatureSet[key]);
			if (m_error) {
				// it_error = true;
				lastError() << " in output number: " << key;
			}
		}
		// if (it_error) m_error = true;
		if (!m_errorQueue.empty()) {
			setValueError("Error while converting FeatureSet.",m_strict);
		} 
		return rFeatureSet;
	}

	/// Convert Python list (backward compatibility)
	if (PyList_Check(pyValue)) {
		
		PyObject *pyFeatureList; // This will be borrowed reference

		//Parse Output List for each element (FeatureSet)
		m_error = false;
		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyValue); ++i) {
			//Get i-th FeatureList (Borrowed Reference)
			pyFeatureList = PyList_GET_ITEM(pyValue,i);
			PyValue_To_rValue(pyFeatureList,rFeatureSet[i]);
			if (m_error) {
				lastError() << " in output number: " << i;
			}
		}
		if (!m_errorQueue.empty()) m_error = true; 
		return rFeatureSet;
	}

	/// accept None return values
	if (pyValue == Py_None) return rFeatureSet;

	/// give up
	std::string msg = "Unsupported return type. Expected list or vampy.FeatureSet(). ";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_FeatureSet failed. Error: " << msg << endl;
#endif			
	return rFeatureSet;
}

Vamp::RealTime
PyTypeInterface::PyValue_To_RealTime(PyObject* pyValue) const
{
// We accept integer sample counts (for backward compatibility)
// or PyRealTime objects and convert them to Vamp::RealTime
	
	if (PyRealTime_CheckExact(pyValue))
	{
		/// just create a copy of the wrapped object
		return Vamp::RealTime(*PyRealTime_AS_REALTIME(pyValue));
	}

	// assume integer sample count
	long sampleCount = m_conv.PyValue_To_Long(pyValue);
	if (m_conv.error) {
		std::string msg = "Unexpected value passed as RealTime.\nMust be vampy.RealTime type or integer sample count.";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_To_RealTime failed. " << msg << endl;
#endif		
		return Vamp::RealTime();
	}

#ifdef _DEBUG_VALUES
	Vamp::RealTime rt = 
		Vamp::RealTime::frame2RealTime(sampleCount,m_inputSampleRate );
	cerr << "RealTime: " << (long)sampleCount << ", ->" << rt.toString() << endl;
	return rt;
#else
	return Vamp::RealTime::frame2RealTime(sampleCount,m_inputSampleRate );	
#endif

}

Vamp::Plugin::OutputDescriptor::SampleType 
PyTypeInterface::PyValue_To_SampleType(PyObject* pyValue) const
{
	/// convert simulated enum values 
	/// { OneSamplePerStep,FixedSampleRate,VariableSampleRate }
	if (PyInt_CheckExact(pyValue)) {
		long lst = PyInt_AS_LONG(pyValue);
		if (lst<0 || lst>2) {
			setValueError("Overflow error. SampleType has to be one of { OneSamplePerStep,FixedSampleRate,VariableSampleRate }\n(an integer in the range of 0..2) or a string value naming the type.",m_strict);
			return Vamp::Plugin::OutputDescriptor::SampleType();
		}
		return (Vamp::Plugin::OutputDescriptor::SampleType) lst; 
	}
	
	/// convert string (backward compatible)
	if (PyString_CheckExact(pyValue)) {
		Vamp::Plugin::OutputDescriptor::SampleType st;
		st = (Vamp::Plugin::OutputDescriptor::SampleType) sampleKeys[m_conv.PyValue_To_String(pyValue)]; 
		if (m_conv.error) {
			std::string msg = "Unexpected value passed as SampleType. Must be one of { OneSamplePerStep,FixedSampleRate,VariableSampleRate }\n(an integer in the range of 0..2) or a string value naming the type.";
			setValueError(msg,m_strict);
			return Vamp::Plugin::OutputDescriptor::SampleType();
		}
		return st;
	}

	/// give up
	std::string msg = "Unsupported return type. Expected one of { OneSamplePerStep,FixedSampleRate,VariableSampleRate }\n(an integer in the range of 0..2) or a string value naming the type.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_SampleType failed. Error: " << msg << endl;
#endif			
	return Vamp::Plugin::OutputDescriptor::SampleType();
}

Vamp::Plugin::InputDomain 
PyTypeInterface::PyValue_To_InputDomain(PyObject* pyValue) const
{
	/// convert simulated enum values { TimeDomain,FrequencyDomain }
	if (PyInt_CheckExact(pyValue)) {
		long lst = PyInt_AS_LONG(pyValue);
		if (lst!=0 && lst!=1) {
			setValueError("Overflow error. InputDomain has to be one of { TimeDomain,FrequencyDomain }\n(an integer in the range of 0..1) or a string value naming the type.",m_strict);
			return Vamp::Plugin::InputDomain();
		}
		return (Vamp::Plugin::InputDomain) lst; 
	}
	
	/// convert string (backward compatible)
	if (PyString_CheckExact(pyValue)) {
		Vamp::Plugin::InputDomain id;
		id = (m_conv.PyValue_To_String(pyValue) == "FrequencyDomain")?Vamp::Plugin::FrequencyDomain:Vamp::Plugin::TimeDomain;
		if (m_conv.error) 
		{
			std::string msg = "Unexpected value passed as SampleType. Must be one of { TimeDomain,FrequencyDomain }\n(an integer in the range of 0..1) or a string value naming the type.";
			setValueError(msg,m_strict);
			return Vamp::Plugin::InputDomain();
		}
		return id;
	}

	/// give up
	std::string msg = "Unsupported return type. Expected one of { TimeDomain,FrequencyDomain }\n(an integer in the range of 0..1) or a string value naming the type.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_InputDomain failed. Error: " << msg << endl;
#endif			
	return Vamp::Plugin::InputDomain();
}

/* Convert Sample Buffers to Python */

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

/// OutputDescriptor
void
PyTypeInterface::SetValue(Vamp::Plugin::OutputDescriptor& od, std::string& key, PyObject* pyValue) const
{
	switch (outKeys[key])
	{
		case o::not_found:
			setValueError("Unknown key in Vamp OutputDescriptor",m_strict);
			cerr << "Unknown key in Vamp OutputDescriptor: " << key << endl;
			break;
			case o::identifier: 
			_convert(pyValue,od.identifier);
			break;				
		case o::name: 			
			_convert(pyValue,od.name);
			break;
		case o::description:
			_convert(pyValue,od.description);
			break;
		case o::unit:
			_convert(pyValue,od.unit);
			break;
		case o::hasFixedBinCount:
			_convert(pyValue,od.hasFixedBinCount);
			break;
		case o::binCount:
			_convert(pyValue,od.binCount);
			break;
		case o::binNames:
			_convert(pyValue,od.binNames);
			break;
		case o::hasKnownExtents:
			_convert(pyValue,od.hasKnownExtents);
			break;
		case o::minValue:
			_convert(pyValue,od.minValue);
			break;
		case o::maxValue:
			_convert(pyValue,od.maxValue);
			break;
		case o::isQuantized:
			_convert(pyValue,od.isQuantized);
			break;					
		case o::quantizeStep:
			_convert(pyValue,od.quantizeStep);
			break;
		case o::sampleType:
			_convert(pyValue,od.sampleType);
			break;
		case o::sampleRate:
			_convert(pyValue,od.sampleRate);
			break;
		case o::hasDuration:
			_convert(pyValue,od.hasDuration);
			break;
		default:
			setValueError("Unknown key in Vamp OutputDescriptor",m_strict);
			cerr << "Invalid key in Vamp OutputDescriptor: " << key << endl;
	}
}

/// ParameterDescriptor
void
PyTypeInterface::SetValue(Vamp::Plugin::ParameterDescriptor& pd, std::string& key, PyObject* pyValue) const
{
	switch (parmKeys[key]) 
	{
		case p::not_found :
			setValueError("Unknown key in Vamp ParameterDescriptor",m_strict);
			cerr << "Unknown key in Vamp ParameterDescriptor: " << key << endl; 
			break;
		case p::identifier:
			_convert(pyValue,pd.identifier);
			break;				
		case p::name:
			_convert(pyValue,pd.name);
			break;
		case p::description: 	
			_convert(pyValue,pd.description);
			break; 								
		case p::unit:
			_convert(pyValue,pd.unit);
			break; 																		
		case p::minValue:	
			_convert(pyValue,pd.minValue);
			break;
		case p::maxValue:
			_convert(pyValue,pd.maxValue);
			break;
		case p::defaultValue:
			_convert(pyValue,pd.defaultValue);
			break;
		case p::isQuantized:
			_convert(pyValue,pd.isQuantized);
			break;									
		case p::quantizeStep:
			_convert(pyValue,pd.quantizeStep);
			break;
		case p::valueNames:
			_convert(pyValue,pd.valueNames);
			break;
		default :
		 	setValueError("Unknown key in Vamp ParameterDescriptor",m_strict);
			cerr << "Invalid key in Vamp ParameterDescriptor: " << key << endl; 
	}
}

/// Feature (it's like a Descriptor)
bool
PyTypeInterface::SetValue(Vamp::Plugin::Feature& feature, std::string& key, PyObject* pyValue) const
{
	bool found = true;
	switch (ffKeys[key])
	{
		case unknown :
			setValueError("Unknown key in Vamp Feature",m_strict);
			cerr << "Unknown key in Vamp Feature: " << key << endl; 
			found = false;
			break;
		case hasTimestamp:
			_convert(pyValue,feature.hasTimestamp);
			break;				
		case timestamp:
			_convert(pyValue,feature.timestamp);
			break;
		case hasDuration: 	
			_convert(pyValue,feature.hasDuration);
			break;
		case duration:
			_convert(pyValue,feature.duration);
			break;
		case values:
			_convert(pyValue,feature.values);
			break; 								
		case label:
			_convert(pyValue,feature.label);
			break;
		default:
			setValueError("Unknown key in Vamp Feature",m_strict);
			found = false;
	}
	return found;
}


/* Error handling */

void
PyTypeInterface::setValueError (std::string message, bool strict) const
{
	m_error = true;
	m_errorQueue.push(ValueError(message,strict));
}

/// return a reference to the last error or creates a new one.
ValueError&
PyTypeInterface::lastError() const 
{
	m_error = false;
	if (!m_errorQueue.empty()) return m_errorQueue.back();
	else {
		m_errorQueue.push(ValueError("Type conversion error.",m_strict));
		return m_errorQueue.back();
	}
}

/// helper function to iterate over the error message queue:
/// pops the oldest item
ValueError 
PyTypeInterface::getError() const
{
	if (!m_errorQueue.empty()) {
		ValueError e = m_errorQueue.front();
		m_errorQueue.pop();
		if (m_errorQueue.empty()) m_error = false;
		return e;
	}
	else {
		m_error = false;
		return ValueError();
	}
}

/* Utilities */

bool
PyTypeInterface::initMaps() const
{

	if (isMapInitialised) return true;

	outKeys["identifier"] = o::identifier;
	outKeys["name"] = o::name;
	outKeys["description"] = o::description;
	outKeys["unit"] = o::unit;
	outKeys["hasFixedBinCount"] = o::hasFixedBinCount; 
	outKeys["binCount"] = o::binCount;
	outKeys["binNames"] = o::binNames;
	outKeys["hasKnownExtents"] = o::hasKnownExtents;
	outKeys["minValue"] = o::minValue;
	outKeys["maxValue"] = o::maxValue;
	outKeys["isQuantized"] = o::isQuantized;
	outKeys["quantizeStep"] = o::quantizeStep;
	outKeys["sampleType"] = o::sampleType;
	outKeys["sampleRate"] = o::sampleRate;
	outKeys["hasDuration"] = o::hasDuration;

	sampleKeys["OneSamplePerStep"] = OneSamplePerStep;
	sampleKeys["FixedSampleRate"] = FixedSampleRate;
	sampleKeys["VariableSampleRate"] = VariableSampleRate;

	ffKeys["hasTimestamp"] = hasTimestamp;
	ffKeys["timestamp"] = timestamp; // this is the correct one
	ffKeys["timeStamp"] = timestamp; // backward compatible
	ffKeys["hasDuration"] = hasDuration;
	ffKeys["duration"] = duration;
	ffKeys["values"] = values;
	ffKeys["label"] = label;

	parmKeys["identifier"] = p::identifier;
	parmKeys["name"] = p::name;
	parmKeys["description"] = p::description;
	parmKeys["unit"] = p::unit;
	parmKeys["minValue"] = p::minValue;
	parmKeys["maxValue"] = p::maxValue;
	parmKeys["defaultValue"] = p::defaultValue;
	parmKeys["isQuantized"] = p::isQuantized;
	parmKeys["quantizeStep"] = p::quantizeStep;
	parmKeys["valueNames"] = p::valueNames;

	isMapInitialised = true;
	return true;
}
