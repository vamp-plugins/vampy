/* -*- c-basic-offset: 8 indent-tabs-mode: t -*- */
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
 * This Vamp plugin is a wrapper for Python Scripts. (VamPy)
 * Centre for Digital Music, Queen Mary, University of London.
 * Copyright 2008, George Fazekas.

TODO:	needs more complete error checking 
	  	needs correct implementation of Python threading
	  	more efficient data conversion using the buffering interface or ctypes
	 	Vamp programs not implemented
		support multiple plugins per script in scanner
		ensure proper cleanup, host do a good job though

*/

#include <Python.h>
#include "PyPlugin.h"
#include "PyTypeInterface.h"
#include <stdlib.h>
#include "PyExtensionModule.h"
//#include "PyRealTime.h"


#ifdef _WIN32
#define PATHSEP ('\\')
#else 
#define PATHSEP ('/')
#endif

//#define _DEBUG

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::map;

Mutex PyPlugin::m_pythonInterpreterMutex;

PyPlugin::PyPlugin(std::string pluginKey, float inputSampleRate, PyObject *pyClass, int &instcount) :
	Plugin(inputSampleRate),
	m_pyClass(pyClass),
	m_instcount(instcount),
	m_stepSize(0),
	m_blockSize(0),
	m_channels(0),
	m_plugin(pluginKey),
	m_class(pluginKey.substr(pluginKey.rfind(':')+1,pluginKey.size()-1)),
	m_path((pluginKey.substr(0,pluginKey.rfind(PATHSEP)))),
	m_processType(0),
	m_pyProcess(NULL),
	m_inputDomain(TimeDomain),
	m_quitOnErrorFlag(false),
	m_debugFlag(false)
{	
	m_ti.setInputSampleRate(inputSampleRate);
	MutexLocker locker(&m_pythonInterpreterMutex);
	cerr << "Creating instance " << m_instcount << " of " << pluginKey << endl;

	if (m_instcount == 0) initvampy();
	m_instcount++;
	
	// if (!PyImport_ImportModule("vampy"))
	// 	cerr << "Could not import extension." << endl;

	// Create an instance
	Py_INCREF(m_pyClass);
	PyObject *pyInputSampleRate = PyFloat_FromDouble(inputSampleRate);
	PyObject *args = PyTuple_Pack(1, pyInputSampleRate);
	m_pyInstance = PyObject_Call(m_pyClass, args, NULL);
	
	if (!m_pyInstance || PyErr_Occurred()) { 
		if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
		Py_DECREF(m_pyClass);
		Py_CLEAR(args);
		Py_CLEAR(pyInputSampleRate);
		cerr << "PyPlugin::PyPlugin: Failed to create Python plugin instance for key \"" 
		<< pluginKey << "\" (is the 1-arg class constructor from sample rate correctly provided?)" << endl;
		throw std::string("Constructor failed");
	}
	Py_INCREF(m_pyInstance);
	Py_DECREF(args);
	Py_DECREF(pyInputSampleRate);
	
	//query the debug flag
	m_debugFlag = getBooleanFlag("vampy_debug_messages",true);
	if (m_debugFlag) cerr << "Debug messages ON for Vampy plugin: " << m_class << endl;
	else cerr << "Debug messages OFF for Vampy plugin: " << m_class << endl;
	
	//query the quit on error flag
	m_quitOnErrorFlag = getBooleanFlag("quit_on_type_error",false);
	if (m_debugFlag && m_quitOnErrorFlag) cerr << "Quit on type error ON for: " << m_class << endl;
   
	//query the type conversion mode flag
	bool st_flag = getBooleanFlag("use_strict_type_conversion",false);
	if (m_debugFlag && st_flag) cerr << "Strict type conversion ON for: " << m_class << endl;
	m_ti.setStrictTypingFlag(st_flag);
}

PyPlugin::~PyPlugin()
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	m_instcount--;
	cerr << "Deleting plugin instance. Count: " << m_instcount << endl;
	
	if (m_pyInstance) Py_DECREF(m_pyInstance);
	//we increase the class refcount before creating an instance 
	if (m_pyClass) Py_DECREF(m_pyClass); 
	if (m_pyProcess) Py_CLEAR(m_pyProcess);
	if (m_instcount == 0) cleanModule();

#ifdef _DEBUG
	cerr << "PyPlugin::PyPlugin:" << m_class << " instance " << m_instcount << " deleted." << endl;
#endif
}


string
PyPlugin::getIdentifier() const
{	
	MutexLocker locker(&m_pythonInterpreterMutex);
	string rString="vampy-xxx";
	if (!m_debugFlag) return genericMethodCall("getIdentifier",rString);

	rString = genericMethodCall("getIdentifier",rString);
	if (rString == "vampy-xxx")
		cerr << "Warning: Plugin must return a unique identifier." << endl;
	return rString;
}

string
PyPlugin::getName() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	string rString="VamPy Plugin (Noname)";
    return genericMethodCall("getName",rString);
}

string
PyPlugin::getDescription() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	string rString="Not given. (Hint: Implement getDescription method.)";
	return genericMethodCall("getDescription",rString);
}


string
PyPlugin::getMaker() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	string rString="VamPy Plugin.";
	return genericMethodCall("getMaker",rString);
}

int
PyPlugin::getPluginVersion() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	size_t rValue=2;
	return genericMethodCall("getPluginVersion",rValue);
}

string
PyPlugin::getCopyright() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	string rString="Licence information not available.";
	return genericMethodCall("getCopyright",rString);
}


bool
PyPlugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{

	if (channels < getMinChannelCount() ||
	    channels > getMaxChannelCount()) return false;

	m_inputDomain = getInputDomain();

	//Note: placing Mutex before the calls above causes deadlock !!
	MutexLocker locker(&m_pythonInterpreterMutex);

	m_stepSize = stepSize;
	m_blockSize = blockSize;
	m_channels = channels;

	//query the process implementation type
	//two optional flags can be used: 'use_numpy_interface' or 'use_legacy_interface'
	//if they are not provided, we fall back to the original method
	setProcessType();
	
	return genericMethodCallArgs<bool>("initialise",channels,stepSize,blockSize);
}

void
PyPlugin::reset()
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	genericMethodCall("reset");
}

PyPlugin::InputDomain 
PyPlugin::getInputDomain() const  
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);
	// Note: Vamp enum type is mapped to Python string !!
	// Is there a better way? (Enums are not native to Python)
	string rValue = "TimeDomain";
	genericMethodCall("getInputDomain",rValue);
	return (rValue == "FrequencyDomain")?FrequencyDomain:TimeDomain;
}

size_t 
PyPlugin::getPreferredBlockSize() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);
	size_t rValue = 0;
	return genericMethodCall("getPreferredBlockSize",rValue); 
}

size_t 
PyPlugin::getPreferredStepSize() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);
	size_t rValue = 0;
    return genericMethodCall("getPreferredStepSize",rValue); 
}

size_t 
PyPlugin::getMinChannelCount() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);
	size_t rValue = 1;
    return genericMethodCall("getMinChannelCount",rValue); 
}

size_t 
PyPlugin::getMaxChannelCount() const 
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);
	size_t rValue = 1;
    return genericMethodCall("getMaxChannelCount",rValue); 
}	

PyPlugin::OutputList
PyPlugin::getOutputDescriptors() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	OutputList list;
	return genericMethodCall("getOutputDescriptors",list);
}

PyPlugin::ParameterList
PyPlugin::getParameterDescriptors() const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	ParameterList list;
	///Note: This function is often called first by the host.
	if (!m_pyInstance) {cerr << "Error: pyInstance is NULL" << endl; return list;}
	return genericMethodCall("getParameterDescriptors",list);
}

void PyPlugin::setParameter(std::string paramid, float newval)
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	genericMethodCallArgs<NoneType>("setParameter",paramid,newval);
}

float PyPlugin::getParameter(std::string paramid) const
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	return genericMethodCallArgs<float>("getParameter",paramid);
}

#ifdef _DEBUG
static int proccounter = 0;
#endif

PyPlugin::FeatureSet
PyPlugin::process(const float *const *inputBuffers,Vamp::RealTime timestamp)
{
	MutexLocker locker(&m_pythonInterpreterMutex);

#ifdef _DEBUG	
	cerr << "[call] process, frame:" << proccounter << endl;
	proccounter++;
#endif

    if (m_blockSize == 0 || m_channels == 0) {
	cerr << "ERROR: PyPlugin::process: "
	     << "Plugin has not been initialised" << endl;
	return FeatureSet();
    }

	if (m_processType == not_implemented) {
	cerr << "ERROR: In Python plugin [" << m_class   
		 << "] No process implementation found. Returning empty feature set." << endl;
	return FeatureSet();
	}

	// string method=PyString_AsString(m_pyProcess);

		PyObject *pyOutputList = NULL;
		
		if (m_processType == numpyProcess) {
			pyOutputList = numpyProcessCall(inputBuffers,timestamp);
		} 
		
		if (m_processType == legacyProcess) { 
			pyOutputList = legacyProcessCall(inputBuffers,timestamp);
		}

		FeatureSet rFeatureset;
		rFeatureset = m_ti.PyValue_To_FeatureSet(pyOutputList);
		Py_CLEAR(pyOutputList);
		return rFeatureset;
		
}

PyObject*
PyPlugin::numpyProcessCall(const float *const *inputBuffers,Vamp::RealTime timestamp)
{
	PyObject *pyOutputList = NULL;
	
	//create a list of buffers
	PyObject *pyChannelList = PyList_New((Py_ssize_t) m_channels);
	for (size_t i=0; i < m_channels; ++i) {
		//Expose memory using the Buffer Interface of C/API		
		//This will virtually pass a pointer which can be 
		//recasted in Python code as float or complex array             
		PyObject *pyBuffer = PyBuffer_FromMemory
		((void *) (float *) inputBuffers[i], 
		(Py_ssize_t) sizeof(float) * m_blockSize);

		PyList_SET_ITEM(pyChannelList, (Py_ssize_t) i, pyBuffer);
		}
/*
		//(1) pass RealTime as frameCount
		PyObject *pyLongSample = PyLong_FromLong (
		Vamp::RealTime::realTime2Frame 
		(timestamp, (unsigned int) m_inputSampleRate));
		
		//Call python process (returns new reference)
		pyOutputList = PyObject_CallMethodObjArgs
		(m_pyInstance,m_pyProcess,pyChannelList,pyLongSample,NULL);
 */   	
		//(2) pass RealTime as PyRealTime
		PyObject *pyRealTime = PyRealTime_FromRealTime(timestamp);

		//Call python process (returns new reference)
		pyOutputList = PyObject_CallMethodObjArgs
		(m_pyInstance,m_pyProcess,pyChannelList,pyRealTime,NULL);
					
		Py_DECREF(pyChannelList);
		// Py_DECREF(pyLongSample);
		Py_DECREF(pyRealTime);
		return pyOutputList;
}

PyObject*
PyPlugin::legacyProcessCall(const float *const *inputBuffers,Vamp::RealTime timestamp)
{
	PyObject *pyOutputList = NULL;
	
	//create a list of lists
	PyObject *pyChannelList = PyList_New((Py_ssize_t) m_channels);
	for (size_t i=0; i < m_channels; ++i) {
		//New list object
		PyObject *pyFloat, *pyList;
		pyList = PyList_New((Py_ssize_t) m_blockSize);

		//Pack samples into a Python List Object
		//pyFloat types will always be new references, 
		//these will be discarded when the list is deallocated
		for (size_t j = 0; j < m_blockSize; ++j) {
			pyFloat=PyFloat_FromDouble(
				(double) inputBuffers[i][j]);
			PyList_SET_ITEM(pyList, (Py_ssize_t) j, pyFloat);
		}
		PyList_SET_ITEM(pyChannelList, (Py_ssize_t) i, pyList);				
	}

	//pass RealTime as frameCount
	PyObject *pyLongSample = PyLong_FromLong (
	Vamp::RealTime::realTime2Frame 
	(timestamp, (unsigned int) m_inputSampleRate));

	//Call python process (returns new reference)
	pyOutputList = PyObject_CallMethodObjArgs
	(m_pyInstance,m_pyProcess,pyChannelList,pyLongSample,NULL);

	Py_DECREF(pyChannelList);
	Py_DECREF(pyLongSample);
	return pyOutputList;
}

PyPlugin::FeatureSet
PyPlugin::getRemainingFeatures()
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	FeatureSet rValue;
	return genericMethodCall("getRemainingFeatures",rValue); 
}


bool
PyPlugin::getBooleanFlag(char flagName[], bool defValue = false) const
{
	bool rValue = defValue;
	if (PyObject_HasAttrString(m_pyInstance,flagName))
	{
		PyObject *pyValue = PyObject_GetAttrString(m_pyInstance,flagName);
		if (!pyValue) 
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		} else {
			rValue = m_ti.PyValue_To_Bool(pyValue);
			if (m_ti.error) { 
				cerr << m_ti.lastError().message << endl;
				Py_CLEAR(pyValue);
				rValue = defValue;
			} else Py_DECREF(pyValue);
		}
	}
	if (m_debugFlag) cerr << FLAG_VALUE << endl;
	return rValue;
}

void
PyPlugin::setProcessType()
{
	//quering process implementation type
	char legacyMethod[]="process";
	char numpyMethod[]="processN";

	if (PyObject_HasAttrString(m_pyInstance,legacyMethod) &&
	    m_processType == 0) 
	{ 
		m_processType = legacyProcess;
		m_pyProcess = PyString_FromString(legacyMethod);
	}

	if (PyObject_HasAttrString(m_pyInstance,numpyMethod) &&
	    m_processType == 0)
	{
		m_processType = numpyProcess;
		m_pyProcess = PyString_FromString(numpyMethod);
	}

	// These flags are optional. If provided, they override the
	// implementation type making the use of the odd processN() 
	// function redundant.
	// However, the code above provides backwards compatibility.
	if (getBooleanFlag("use_numpy_interface",false)) 
		m_processType = numpyProcess;
	if (getBooleanFlag("use_legacy_interface",false)) 
		m_processType = legacyProcess;
	if (m_debugFlag && m_processType) 
		cerr << "Process type: " << ((m_processType==numpyProcess)?"numpy process":"legacy process") << endl;
	
	if (!m_processType)
	{
		m_processType = not_implemented;
		m_pyProcess = NULL;
		char method[]="initialise::setProcessType";
		cerr << PLUGIN_ERROR << " No process implementation found. Plugin will do nothing." << endl;
	}
}

