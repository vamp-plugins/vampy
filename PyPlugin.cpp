/*
    Vamp

    An API for audio analysis and feature extraction plugins.

    Centre for Digital Music, Queen Mary, University of London.
    Copyright 2006 Chris Cannam.
  
    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Except as contained in this notice, the names of the Centre for
    Digital Music; Queen Mary, University of London; and Chris Cannam
    shall not be used in advertising or otherwise to promote the sale,
    use or other dealings in this Software without prior written
    authorization.
*/



/**
 * This VAMP plugin is a wrapper for Python Scripts. (vampy)
 * Centre for Digital Music, Queen Mary, University of London.
 * Copyright 2008, George Fazekas.

TODO:	needs more complete error checking 
	  	needs correct implementation of Python threading
	  	more efficient data conversion using the buffering interface or ctypes
	 	VAMP programs not implemented
		support multiple plugins per script in scanner
		ensure proper cleanup, host do a good job though

*/

#include <Python.h>
#include "PyPlugin.h"

#ifdef _WIN32
#define pathsep ('\\')
#else 
#define pathsep ('/')
#endif

#define _DEBUG

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::map;

// Maps to associate strings with enum values
static std::map<std::string, eOutDescriptors> outKeys;
static std::map<std::string, eSampleTypes> sampleKeys;
static std::map<std::string, eFeatureFields> ffKeys;
static std::map<std::string, p::eParmDescriptors> parmKeys;

Mutex PyPlugin::m_pythonInterpreterMutex;

void initMaps()
{
	outKeys["identifier"] = identifier;
	outKeys["name"] = name;
	outKeys["description"] = description;
	outKeys["unit"] = unit;
	outKeys["hasFixedBinCount"] = hasFixedBinCount; 
	outKeys["binCount"] = binCount;
	outKeys["binNames"] = binNames;
	outKeys["hasKnownExtents"] = hasKnownExtents;
	outKeys["minValue"] = minValue;
	outKeys["maxValue"] = maxValue;
	outKeys["isQuantized"] = isQuantized;
	outKeys["quantizeStep"] = quantizeStep;
	outKeys["sampleType"] = sampleType;
	outKeys["sampleRate"] = sampleRate;

	sampleKeys["OneSamplePerStep"] = OneSamplePerStep;
	sampleKeys["FixedSampleRate"] = FixedSampleRate;
	sampleKeys["VariableSampleRate"] = VariableSampleRate;

	ffKeys["hasTimestamp"] = hasTimestamp;
	ffKeys["timeStamp"] = timeStamp;
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

}


//missing API helper: convert Python list to C++ vector of strings
//TODO: these could be templates if we need more of this kind
std::vector<std::string> PyList_To_StringVector (PyObject *inputList) {
	
	std::vector<std::string> Output;
	std::string ListElement;
	PyObject *pyString = NULL;
	
	if (!PyList_Check(inputList)) return Output;

	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(inputList); ++i) {
		//Get next list item (Borrowed Reference)
		pyString = PyList_GET_ITEM(inputList,i);
		ListElement = (string) PyString_AsString(PyObject_Str(pyString));
		Output.push_back(ListElement);
	}
	return Output;
}

//missing API helper: convert Python list to C++ vector of floats
std::vector<float> PyList_As_FloatVector (PyObject *inputList) {
	
	std::vector<float> Output;
	float ListElement;
	PyObject *pyFloat = NULL;
	
	if (!PyList_Check(inputList)) return Output; 

	for (Py_ssize_t k = 0; k < PyList_GET_SIZE(inputList); ++k) {
		//Get next list item (Borrowed Reference)
		pyFloat =  PyList_GET_ITEM(inputList,k);
		ListElement = (float) PyFloat_AS_DOUBLE(pyFloat);
		Output.push_back(ListElement);
	}

	return Output;
}

/* TODO: find out why this produces error, also 
		do sg more clever about handling RealTime
Vamp::RealTime 
PyFrame_As_RealTime (PyObject *frameNo,size_t inputSampleRate) {
Vamp::RealTime result =  
Vamp::RealTime::frame2RealTime((size_t)PyInt_AS_LONG(frameNo), inputSampleRate);
return result;
}
*/


PyPlugin::PyPlugin(std::string pluginKey,float inputSampleRate, PyObject *pyInstance) :
    Plugin(inputSampleRate),
	m_pyInstance(pyInstance),
	m_stepSize(0),
    m_previousSample(0.0f),
	m_plugin(pluginKey),
	m_class(pluginKey.substr(pluginKey.rfind(':')+1,pluginKey.size()-1)),
	m_path((pluginKey.substr(0,pluginKey.rfind(pathsep))))
{	
}

PyPlugin::~PyPlugin()
{
	cerr << "PyPlugin::PyPlugin:" << m_class 
	<< " Instance deleted." << endl;
}


string
PyPlugin::getIdentifier() const
{	
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getIdentifier"; 
	cerr << "[call] " << method << endl;
	string rString="VampPy-x";

	if ( PyObject_HasAttrString(m_pyInstance,method) ) {

		//Call the method
		PyObject *pyString = 
		PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyString_Check(pyString)) {
			Py_CLEAR(pyString);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rString;
		}

		rString=PyString_AsString(pyString);
		Py_CLEAR(pyString);
	}
	cerr << "Warning: Plugin must return a unique identifier." << endl;
	return rString;
}


string
PyPlugin::getName() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getName";
	cerr << "[call] " << method << endl;
	string rString="VamPy Plugin (Noname)";
	
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {

		//Call the method
		PyObject *pyString = 
		PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyString_Check(pyString)) {
			Py_CLEAR(pyString);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rString;
		}

		rString=PyString_AsString(pyString);
		Py_CLEAR(pyString);
	}
    return rString;
}

string
PyPlugin::getDescription() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getDescription";
	cerr << "[call] " << method << endl;
	string rString="Not given. (Hint: Implement getDescription method.)";
	
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {

		//Call the method
		PyObject *pyString = 
		PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyString_Check(pyString)) {
			Py_CLEAR(pyString);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rString;
		}

		rString=PyString_AsString(pyString);
		Py_CLEAR(pyString);
	}
	return rString;
}

string
PyPlugin::getMaker() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getMaker";
	cerr << "[call] " << method << endl;
	string rString="Generic VamPy Plugin.";
	
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {

		//Call the method
		PyObject *pyString = 
		PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyString_Check(pyString)) {
			Py_CLEAR(pyString);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rString;
		}

		rString=PyString_AsString(pyString);
		Py_CLEAR(pyString);
	}
    return rString;
}

int
PyPlugin::getPluginVersion() const
{
    return 2;
}

string
PyPlugin::getCopyright() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getCopyright";
	cerr << "[call] " << method << endl;
	string rString="BSD License";
	
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {

		//Call the method
		PyObject *pyString = 
		PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyString_Check(pyString)) {
			Py_CLEAR(pyString);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rString;
		}

		
		rString=PyString_AsString(pyString);
		Py_CLEAR(pyString);
	}
    return rString;
}


bool
PyPlugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
	MutexLocker locker(&m_pythonInterpreterMutex);

    if (channels < getMinChannelCount() ||
	channels > getMaxChannelCount()) return false;

    m_stepSize = std::min(stepSize, blockSize);
	char method[]="initialise";
	cerr << "[call] " << method << endl;

		//Check if the method is implemented in Python else return false
		if (PyObject_HasAttrString(m_pyInstance,method)) {
   			
			PyObject *pyMethod = PyString_FromString(method);
			PyObject *pyChannels = PyInt_FromSsize_t((Py_ssize_t)channels);
			PyObject *pyStepSize = PyInt_FromSsize_t((Py_ssize_t)m_stepSize);
			PyObject *pyBlockSize = PyInt_FromSsize_t((Py_ssize_t)blockSize);
			PyObject *pyInputSampleRate = PyFloat_FromDouble((double)m_inputSampleRate);
			
			//Call the method
			PyObject *pyBool = 
			PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyChannels,pyStepSize,pyBlockSize,pyInputSampleRate,NULL);
			
			Py_DECREF(pyMethod);
			Py_DECREF(pyChannels);
			Py_DECREF(pyStepSize);
			Py_DECREF(pyBlockSize);
			Py_DECREF(pyInputSampleRate);

			//Check return value
			if (!PyBool_Check(pyBool)) {
				Py_CLEAR(pyBool);
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
				<< "] Expected Bool return value." << endl;
				return false;
			}

			if (pyBool == Py_True) {  
				Py_CLEAR(pyBool); 
				return true;
			} else {
				Py_CLEAR(pyBool); 
				return false;}								
		} 
    	return true;
}

void
PyPlugin::reset()
{
	//!!! implement this!
    m_previousSample = 0.0f;
}

PyPlugin::InputDomain PyPlugin::getInputDomain() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getInputDomain";
	cerr << "[call] " << method << endl;
	PyPlugin::InputDomain rValue = TimeDomain; // TimeDomain
	
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {

		PyObject *pyString = PyObject_CallMethod(m_pyInstance, method, NULL);
		
		//Check return value
		if (!PyString_Check(pyString)) {
			Py_CLEAR(pyString);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rValue;
		}
		
		string domain = (string) PyString_AsString(pyString);
		if (domain == "FrequencyDomain") rValue = FrequencyDomain;
		Py_CLEAR(pyString);
	}
    return rValue; 
}

size_t PyPlugin::getPreferredBlockSize() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getPreferredBlockSize";
	cerr << "[call] " << method << endl;
	size_t rValue=0; //not set by default
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {
		PyObject *pyInt = PyObject_CallMethod(m_pyInstance, method, NULL);
		
		//Check return value
		if (!PyInt_Check(pyInt)) {
			Py_CLEAR(pyInt);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected Integer return value." << endl;
			return rValue;
		}
		
		rValue=(size_t)PyInt_AS_LONG(pyInt);
		Py_CLEAR(pyInt);
	}
    return rValue; 
}

//size_t PyPlugin::getPreferredStepSize() const { return 0; }
size_t PyPlugin::getPreferredStepSize() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getPreferredStepSize";
	cerr << "[call] " << method << endl;
	size_t rValue=0; //not set by default
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {
		PyObject *pyInt = PyObject_CallMethod(m_pyInstance, method, NULL);
		
		//Check return value
		if (!PyInt_Check(pyInt)) {
			Py_CLEAR(pyInt);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected Integer return value." << endl;
			return rValue;
		}
		
		rValue=(size_t)PyInt_AS_LONG(pyInt);
		Py_CLEAR(pyInt);
	}
    return rValue; 
}

size_t PyPlugin::getMinChannelCount() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getMinChannelCount";
	cerr << "[call] " << method << endl;
	size_t rValue=1; //default value
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {
		PyObject *pyInt = PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyInt_Check(pyInt)) {
			Py_CLEAR(pyInt);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rValue;
		}

		rValue=(size_t)PyInt_AS_LONG(pyInt);
		Py_CLEAR(pyInt);
	}
    return rValue; 
}

size_t PyPlugin::getMaxChannelCount() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getMaxChannelCount";	
	cerr << "[call] " << method << endl;
	size_t rValue=1; //default value
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {
		PyObject *pyInt = PyObject_CallMethod(m_pyInstance, method, NULL);

		//Check return value
		if (!PyInt_Check(pyInt)) {
			Py_CLEAR(pyInt);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected String return value." << endl;
			return rValue;
		}

		rValue=(size_t)PyInt_AS_LONG(pyInt);
		Py_CLEAR(pyInt);
	}
    return rValue; 
}


PyPlugin::OutputList
PyPlugin::getOutputDescriptors() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	//PyEval_AcquireThread(newThreadState);
	OutputList list;
	OutputDescriptor od;
	char method[]="getOutputDescriptors";
	cerr << "[call] " << method << endl;

	//Check if the method is implemented in Python
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {
			
		//Call the method: must return list object (new reference)
		PyObject *pyList = 
		PyObject_CallMethod(m_pyInstance,method, NULL);

		//Check return type
		if (! PyList_Check(pyList) ) {
			Py_CLEAR(pyList);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected List return type." << endl;
			return list;
		}

		//These will all be borrowed references (no need to DECREF)
		PyObject *pyDict, *pyKey, *pyValue;
		
		//Parse Output List
		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
	
			//Get i-th VAMP output descriptor (Borrowed Reference)
			pyDict = PyList_GET_ITEM(pyList,i);
			
			//We only care about dictionaries holding output descriptors
			if ( !PyDict_Check(pyDict) ) continue;
			
			Py_ssize_t pyPos = NULL;
			initMaps();

			//Python Sequence Iterator
			while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue)) 
			{		
				switch (outKeys[PyString_AsString(pyKey)]) 
				{
					case not_found : 	
						cerr << "Unknown key in VAMP OutputDescriptor: " << PyString_AsString(pyKey) << endl; 
						break;
					case identifier: 	
						od.identifier = PyString_AsString(pyValue); 
						break;				
					case name: 			
						od.name = PyString_AsString(pyValue); 
						break;
					case description: 	
						od.description = PyString_AsString(pyValue); 
						break; 								
					case unit: 			
						od.unit = PyString_AsString(pyValue); 
						break; 													
					case hasFixedBinCount:
						od.hasFixedBinCount = (bool) PyInt_AS_LONG(pyValue); 
						break;
					case binCount:
						od.binCount = (size_t) PyInt_AS_LONG(pyValue);
						break;
					case binNames:
						od.binNames = PyList_To_StringVector(pyValue);
						break;
					case hasKnownExtents:
						od.hasKnownExtents = (bool) PyInt_AS_LONG(pyValue); 
						break;					
					case minValue:
						od.minValue = (float) PyFloat_AS_DOUBLE(pyValue);
						break;
					case maxValue:
						od.maxValue = (float) PyFloat_AS_DOUBLE(pyValue);
						break;
					case isQuantized:
						od.isQuantized = (bool) PyInt_AS_LONG(pyValue); 
						break;					
					case quantizeStep:
						od.quantizeStep = (float) PyFloat_AS_DOUBLE(pyValue);
						break;
					case sampleType: 				
						od.sampleType = (OutputDescriptor::SampleType) sampleKeys[PyString_AsString(pyValue)];
						break;
					case sampleRate:
						od.sampleRate = (float) PyFloat_AS_DOUBLE(pyValue);
						break;					
					default : 	
						cerr << "Invalid key in VAMP OutputDescriptor: " << PyString_AsString(pyKey) << endl; 
				} // switch					
			} // while dict keys
			list.push_back(od);
		} // for each memeber in outputlist
		Py_CLEAR(pyList);
	} // if method is implemented
    //PyEval_ReleaseThread(newThreadState);
	return list;
}

PyPlugin::ParameterList
PyPlugin::getParameterDescriptors() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	ParameterList list;
	ParameterDescriptor pd;
	char method[]="getParameterDescriptors";
	cerr << "[call] " << method << endl;

	//Check if the method is implemented in Python
	if ( PyObject_HasAttrString(m_pyInstance,method) ) {
			
		//Call the method: must return list object (new reference)
		PyObject *pyList = 
		PyObject_CallMethod(m_pyInstance,method, NULL);

		//Check return type
		if (! PyList_Check(pyList) ) {
			Py_CLEAR(pyList);
			cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
			<< "] Expected List return type." << endl;
			return list;
		}


		//These will all be borrowed references (no need to DECREF)
		PyObject *pyDict, *pyKey, *pyValue;
		
		//Parse Output List
		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
	
			//Get i-th VAMP output descriptor (Borrowed Reference)
			pyDict = PyList_GET_ITEM(pyList,i);
			
			//We only care about dictionaries holding output descriptors
			if ( !PyDict_Check(pyDict) ) continue;
			
			Py_ssize_t pyPos = NULL;
			initMaps();

			//Python Sequence Iterator
			while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue)) 
			{		
				switch (parmKeys[PyString_AsString(pyKey)]) 
				{
					case not_found : 	
						cerr << "Unknown key in VAMP OutputDescriptor: " << PyString_AsString(pyKey) << endl; 
						break;
					case p::identifier: 	
						pd.identifier = PyString_AsString(pyValue); 
						break;				
					case p::name: 			
						pd.name = PyString_AsString(pyValue); 
						break;
					case p::description: 	
						pd.description = PyString_AsString(pyValue); 
						break; 								
					case p::unit: 			
						pd.unit = PyString_AsString(pyValue); 
						break; 																		
					case p::minValue:
						pd.minValue = (float) PyFloat_AS_DOUBLE(pyValue);
						break;
					case p::maxValue:
						pd.maxValue = (float) PyFloat_AS_DOUBLE(pyValue);
						break;
					case p::defaultValue:
						pd.defaultValue = (float) PyFloat_AS_DOUBLE(pyValue);
						break;
					case p::isQuantized:
						pd.isQuantized = (bool) PyInt_AS_LONG(pyValue); 
						break;					
					default : 	
						cerr << "Invalid key in VAMP OutputDescriptor: " << PyString_AsString(pyKey) << endl; 
				} // switch					
			} // while dict keys
			list.push_back(pd);
		} // for each memeber in outputlist
		Py_CLEAR(pyList);
	} // if 
	return list;
}

void PyPlugin::setParameter(std::string paramid, float newval)
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="setParameter";
	cerr << "[call] " << method << endl;

		//Check if the method is implemented in Python
		if (PyObject_HasAttrString(m_pyInstance,method)) {
   			
			PyObject *pyMethod = PyString_FromString(method);
			PyObject *pyParamid = PyString_FromString(paramid.c_str());
			PyObject *pyNewval = PyFloat_FromDouble((double)newval);
			
			//Call the method
			PyObject *pyBool = 
			PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyParamid,pyNewval,NULL);

			//This only happens if there is a syntax error or so
			if (pyBool == NULL) {
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
				<< "] Error setting parameter: " << paramid << endl;
				if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			}

			Py_DECREF(pyMethod);
			Py_DECREF(pyParamid);
			Py_DECREF(pyNewval);
		}
}

float PyPlugin::getParameter(std::string paramid) const
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	char method[]="getParameter";
	cerr << "[call] " << method << endl;
	float rValue = 0.0f;
	
		//Check if the method is implemented in Python
		if (PyObject_HasAttrString(m_pyInstance,method)) {
   			
			PyObject *pyMethod = PyString_FromString(method);
			PyObject *pyParamid = PyString_FromString(paramid.c_str());
			
			//Call the method
			PyObject *pyFloat = 
			PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyParamid,NULL);

			//Check return type
			if (! PyFloat_Check(pyFloat) ) {
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
					<< "] Expected Float return type." << endl;
				if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
				Py_CLEAR(pyFloat);
				return rValue;
			}
	
			rValue = (float) PyFloat_AS_DOUBLE(pyFloat); 
			
			Py_DECREF(pyMethod);
			Py_DECREF(pyParamid);
			Py_DECREF(pyFloat);
		}

    return rValue;
}

#ifdef _DEBUG
static int proccounter = 0;
#endif

PyPlugin::FeatureSet
PyPlugin::process(const float *const *inputBuffers,
                      Vamp::RealTime timestamp)
{
	MutexLocker locker(&m_pythonInterpreterMutex);

    if (m_stepSize == 0) {
	cerr << "ERROR: PyPlugin::process: "
	     << "Plugin has not been initialised" << endl;
	return FeatureSet();
    }
	static char method[]="process";

#ifdef _DEBUG	
	cerr << "[call] " << method << " frame:" << proccounter << endl;
	proccounter++;
	//cerr << "C: proc..." << proccounter << " " << endl;
#endif

	//proces::Check if the method is implemented in Python
	if ( PyObject_HasAttrString(m_pyInstance,method) )
	{	
		
		//Pack method name into Python Object
		PyObject *pyMethod = PyString_FromString(method);

		//Declare new list object
		PyObject *pyFloat, *pyList;
		pyList = PyList_New((Py_ssize_t) m_stepSize);

		//Pack samples into a Python List Object
		//pyFloat will always be new references, 
		//these will be discarded when the list is deallocated
		for (size_t i = 0; i < m_stepSize; ++i) {
		pyFloat=PyFloat_FromDouble((double) inputBuffers[0][i]);
		PyList_SET_ITEM(pyList, (Py_ssize_t) i, pyFloat);
		}
		
		//Call python process (returns new reference)
		PyObject *pyOutputList = 
		PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyList,NULL);

		//Check return type
		if (pyOutputList == NULL || !PyList_Check(pyOutputList) ) {
			if (pyOutputList == NULL) {				
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
				<< "] Unexpected result." << endl;
				if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			} else {
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
				<< "] Expected List return type." << endl;				
			}
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyOutputList);
			return FeatureSet();
		}
			
		Py_DECREF(pyMethod);
		// Py_DECREF(pyList); 
		// This appears to be tracked by the cyclic garbage collector
		// hence decrefing produces GC error
#ifdef _DEBUG								
		cerr << "Process Returned Features" << endl;
#endif
		// These will ALL be borrowed references
		PyObject *pyFeatureList, *pyDict, *pyKey, *pyValue;

		FeatureSet returnFeatures;

		//Parse Output List for each element (FeatureSet)
		for (Py_ssize_t i = 0; 
					i < PyList_GET_SIZE(pyOutputList); ++i) {
			//cerr << "output (FeatureSet): " << i << endl; 

			//Get i-th FeatureList (Borrowed Reference)
			pyFeatureList = PyList_GET_ITEM(pyOutputList,i);

			//Parse FeatureList for each element (Feature)
			for (Py_ssize_t j = 0; j < PyList_GET_SIZE(pyFeatureList); ++j) {				
				//cerr << "element (FeatureList): " << j << endl; 

				//Get j-th Feature (Borrowed Reference)
				pyDict = PyList_GET_ITEM(pyFeatureList,j);

				//We only care about dictionaries holding a Feature struct
				if ( !PyDict_Check(pyDict) ) continue;

				Py_ssize_t pyPos = NULL;
				bool emptyFeature = true;
				Feature feature;

				//process::Python Sequence Iterator for dictionary
				while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue)) 
				{	
					emptyFeature = false;
					switch (ffKeys[PyString_AsString(pyKey)]) 
					{
						case not_found : 	
							cerr << "Unknown key in VAMP FeatureSet: " 
							<< PyString_AsString(pyKey) << endl; 
							break;
						case hasTimestamp: 	
							feature.hasTimestamp = (bool) PyInt_AS_LONG(pyValue); 
							break;				
						case timeStamp: 			
							feature.timestamp = timestamp + 
							Vamp::RealTime::frame2RealTime
							((size_t)PyInt_AS_LONG(pyValue), (size_t)m_inputSampleRate);
							break;
						case values: 	
							feature.values = PyList_As_FloatVector(pyValue); 
							break; 								
						case label: 			
							feature.label = PyString_AsString(pyValue); 
							break; 													
						default : 	
							cerr << "Invalid key in VAMP FeatureSet: " 
							<< PyString_AsString(pyKey) << endl; 
					} // switch					

				} // while 
				if (emptyFeature) cerr << "Warning: This feature is empty or badly formatted." << endl;
				else returnFeatures[i].push_back(feature);

			}// for j = FeatureList			

		}//for i = FeatureSet
		Py_CLEAR(pyOutputList);
		return returnFeatures;

	}//if (has_attribute)
	return FeatureSet(); 
}



PyPlugin::FeatureSet
PyPlugin::getRemainingFeatures()
{
	MutexLocker locker(&m_pythonInterpreterMutex);

	static char method[]="getRemainingFeatures";
	cerr << "[call] " << method << endl;

	//check if the method is implemented
	if ( ! PyObject_HasAttrString(m_pyInstance,method) ) {
		return FeatureSet(); 
		}
			
		PyObject *pyMethod = PyString_FromString(method);		

		PyObject *pyOutputList = 
		PyObject_CallMethod(m_pyInstance,method, NULL);
					
		//Check return type
		if (pyOutputList == NULL || !PyList_Check(pyOutputList) ) {
			if (pyOutputList == NULL) {				
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
				<< "] Unexpected result." << endl;
				if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			} else {
				cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
				<< "] Expected List return type." << endl;				
			}
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyOutputList);
			return FeatureSet();
		}
		Py_DECREF(pyMethod);
	
		PyObject *pyFeatureList, *pyDict, *pyKey, *pyValue;
		FeatureSet returnFeatures;

		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyOutputList); ++i) {

			pyFeatureList = PyList_GET_ITEM(pyOutputList,i);

			for (Py_ssize_t j = 0; j < PyList_GET_SIZE(pyFeatureList); ++j) {				

				pyDict = PyList_GET_ITEM(pyFeatureList,j);

				if ( !PyDict_Check(pyDict) ) continue;

				Py_ssize_t pyPos = NULL;
				bool emptyFeature = true;
				Feature feature;

				while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue)) 
				{	
					emptyFeature = false;
					switch (ffKeys[PyString_AsString(pyKey)]) 
					{
						case not_found : 	
							cerr << "Unknown key in VAMP FeatureSet: " 
							<< PyString_AsString(pyKey) << endl; 
							break;
						case hasTimestamp: 	
							feature.hasTimestamp = (bool) PyInt_AS_LONG(pyValue); 
							break;				
						// TODO: clarify what to do here
						case timeStamp: 			
							feature.timestamp =  
							Vamp::RealTime::frame2RealTime
							((size_t)PyInt_AS_LONG(pyValue), (size_t)m_inputSampleRate);
							break;
						case values: 	
							feature.values = PyList_As_FloatVector(pyValue); 
							break; 								
						case label: 			
							feature.label = PyString_AsString(pyValue); 
							break; 													
					} // switch					
				} // while 
				if (emptyFeature) cerr << "Warning: This feature is empty or badly formatted." << endl;
				else returnFeatures[i].push_back(feature);
			}// for j 			
		}//for i 
		Py_CLEAR(pyOutputList);
		return returnFeatures;
}

