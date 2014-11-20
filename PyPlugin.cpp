/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "PyPlugin.h"
#include "PyTypeInterface.h"
#include <stdlib.h>
#include "PyExtensionModule.h"
#include "Debug.h"

#ifdef _WIN32
#define PATHSEP ('\\')
#else 
#define PATHSEP ('/')
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::map;

Mutex PyPlugin::m_pythonInterpreterMutex;

PyPlugin::PyPlugin(std::string pluginKey, float inputSampleRate, PyObject *pyClass, int &instcount, bool &numpyInstalled) :
	Plugin(inputSampleRate),
	m_pyClass(pyClass),
	m_instcount(instcount),
	m_stepSize(0),
	m_blockSize(0),
	m_channels(0),
	m_plugin(pluginKey),
	m_class(pluginKey.substr(pluginKey.rfind(':')+1,pluginKey.size()-1)),
	m_path((pluginKey.substr(0,pluginKey.rfind(PATHSEP)))),
	m_processType(not_implemented),
	m_pyProcess(NULL),
	m_inputDomain(TimeDomain),
	m_quitOnErrorFlag(false),
	m_debugFlag(false),
	m_numpyInstalled(numpyInstalled),
	m_processFailure(false)
{	
	m_ti.setInputSampleRate(inputSampleRate);
	MutexLocker locker(&m_pythonInterpreterMutex);
	DSTREAM << "Creating instance " << m_instcount << " of " << pluginKey << endl;
		
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
	
	m_instcount++;
	
	// query and decode vampy flags
	m_vampyFlags = getBinaryFlags("vampy_flags",vf_NULL);

	m_debugFlag = (bool) (m_vampyFlags & vf_DEBUG);
	m_quitOnErrorFlag = (bool) (m_vampyFlags & vf_QUIT);
	bool st_flag = (bool) (m_vampyFlags & vf_STRICT);
	m_useRealTimeFlag = (bool) (m_vampyFlags & vf_REALTIME);
		
	if (m_debugFlag) cerr << "Debug messages ON for Vampy plugin: " << m_class << endl;
	else DSTREAM << "Debug messages OFF for Vampy plugin: " << m_class << endl;
	
	if (m_debugFlag && m_quitOnErrorFlag) cerr << "Quit on type error ON for: " << m_class << endl;
   
	if (m_debugFlag && st_flag) cerr << "Strict type conversion ON for: " << m_class << endl;

	m_ti.setStrictTypingFlag(st_flag);
	m_tc.setStrictTypingFlag(st_flag);

	m_ti.setNumpyInstalled(m_numpyInstalled);
	m_tc.setNumpyInstalled(m_numpyInstalled);

}

PyPlugin::~PyPlugin()
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	m_instcount--;
	// cerr << "Deleting plugin instance. Count: " << m_instcount << endl;
	
	if (m_pyInstance) Py_DECREF(m_pyInstance);
	//we increase the class refcount before creating an instance 
	if (m_pyClass) Py_DECREF(m_pyClass); 
	if (m_pyProcess) Py_CLEAR(m_pyProcess);

	DSTREAM << "PyPlugin::PyPlugin:" << m_class << " instance " << m_instcount << " deleted." << endl;
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
	m_processFailure = false;
	genericMethodCall("reset");
}

PyPlugin::InputDomain 
PyPlugin::getInputDomain() const  
{ 
	MutexLocker locker(&m_pythonInterpreterMutex);
	return genericMethodCall("getInputDomain",m_inputDomain);
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
#ifdef _DEBUG	
	///Note: This function is often called first by the host.
	if (!m_pyInstance) {cerr << "Error: pyInstance is NULL" << endl; return list;}
#endif

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

#ifdef _DEBUG_VALUES
static int proccounter = 0;
#endif

PyPlugin::FeatureSet
PyPlugin::process(const float *const *inputBuffers,Vamp::RealTime timestamp)
{
	MutexLocker locker(&m_pythonInterpreterMutex);

#ifdef _DEBUG_VALUES
	/// we only need this if we'd like to see what frame a set of values belong to 
	cerr << "[Vampy::call] process, frame:" << proccounter << endl;
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
	
	if (m_processFailure) return FeatureSet();
	
	return processMethodCall(inputBuffers,timestamp);

}

PyPlugin::FeatureSet
PyPlugin::getRemainingFeatures()
{
	MutexLocker locker(&m_pythonInterpreterMutex);
	if (m_processFailure) return FeatureSet();
	FeatureSet rValue;
	return genericMethodCall("getRemainingFeatures",rValue); 
}

bool
PyPlugin::getBooleanFlag(const char flagName[], bool defValue = false) const
{
	bool rValue = defValue;
	if (PyObject_HasAttrString(m_pyInstance,flagName))
	{
		PyObject *pyValue = PyObject_GetAttrString(m_pyInstance,flagName);
		if (!pyValue) 
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		} else {
			rValue = m_tc.PyValue_To_Bool(pyValue);
			if (m_tc.error) { 
				Py_CLEAR(pyValue);
				typeErrorHandler(flagName);
				rValue = defValue;
			} else Py_DECREF(pyValue);
		}
	}
	if (m_debugFlag) cerr << FLAG_VALUE << endl;
	return rValue;
}

int
PyPlugin::getBinaryFlags(const char flagName[], eVampyFlags defValue = vf_NULL) const
{
	int rValue = defValue;
	if (PyObject_HasAttrString(m_pyInstance,flagName))
	{
		PyObject *pyValue = PyObject_GetAttrString(m_pyInstance,flagName);
		if (!pyValue) 
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		} else {
			rValue |= (int) m_tc.PyValue_To_Size_t(pyValue);
			if (m_tc.error) { 
				Py_CLEAR(pyValue);
				typeErrorHandler(flagName);
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
	m_processFailure = false;

	if (PyObject_HasAttrString(m_pyInstance,legacyMethod) &&
	    m_processType == 0) 
	{ 
		m_processType = legacyProcess;
		m_pyProcess = PyString_FromString(legacyMethod);
		m_pyProcessCallable = PyObject_GetAttr(m_pyInstance,m_pyProcess);
	}

	if (PyObject_HasAttrString(m_pyInstance,numpyMethod) &&
	    m_processType == 0)
	{
		m_processType = numpy_bufferProcess;
		m_pyProcess = PyString_FromString(numpyMethod);
		m_pyProcessCallable = PyObject_GetAttr(m_pyInstance,m_pyProcess);
	}

	// These flags are optional. If provided, they override the
	// implementation type making the use of the odd processN() 
	// function redundant.
	// However, the code above provides backward compatibility.

	if (m_vampyFlags & vf_BUFFER) {
		m_processType = numpy_bufferProcess;
		if (m_debugFlag) cerr << "Process using (numpy) buffer interface." << endl;
	}

    if (m_vampyFlags & vf_ARRAY) {
#ifdef HAVE_NUMPY
		if (m_numpyInstalled) { m_processType = numpy_arrayProcess;
			if (m_debugFlag) 
				cerr << "Process using numpy array interface." << endl;
		}
		else {
			m_processFailure = true;
			char method[]="initialise::setProcessType";
			cerr << PLUGIN_ERROR
			<< "This plugin requests the Numpy array interface by setting "
			<< " the vf_ARRAY flag in its __init__() function." << endl 
			<< "However, we could not found a version of Numpy compatible with this build of Vampy." << endl
			<< "If you have a numerical library installed that supports the buffer interface, " << endl
			<< "you can request this interface instead by setting the vf_BUFFER flag." << endl;
		}
#else
		m_processFailure = true;
		char method[]="initialise::setProcessType";
		cerr << PLUGIN_ERROR
		<< "Error: This version of vampy was compiled without numpy support, "
		<< "however the vf_ARRAY flag is set for plugin: " << m_class << endl
		<< "The default behaviour is: passing a python list of samples for each channel in process() "
		<< "or a list of memory buffers in processN(). " << endl 
		<< "This can be used create numpy arrays using the numpy.frombuffer() command." << endl;
#endif		
	}
	
	if (!m_pyProcessCallable)
	{
		m_processType = not_implemented;
		m_pyProcess = NULL;
		char method[]="initialise::setProcessType";
		cerr << PLUGIN_ERROR << " No process implementation found. Plugin will do nothing." << endl;
		m_processFailure = true;
	}
}

void
PyPlugin::typeErrorHandler(const char *method, bool process) const
{
	bool strict = false;
	while (m_tc.error || m_ti.error) {
	    ValueError e;
	    if (m_tc.error) e = m_tc.getError();
	    else e = m_ti.getError();
#ifdef HAVE_NUMPY
		// disable the process completely if numpy types are returned 
		// but a compatible version was not loaded.
		// This is required because if an object is returned from
		// the wrong build, malloc complains about its size
		// (i.e. the interpreter doesn't free it properly)
		// and the process may be leaking.
		// Note: this only happens in the obscure situation when
		// someone forces to return wrong numpy types from an 
		// incompatible version using the buffer interface.
		// In this case the incampatible library is still usable,
		// but manual conversion to python builtins is required.
		// If the ARRAY interface is set but Numpy is not installed
		// the process will be disabled already at initialisation.
		if (process && !m_numpyInstalled && e.str().find("numpy")!=std::string::npos) 
		{
			m_processFailure = true;
			cerr << "Warning: incompatible numpy type encountered. Disabling process." << endl;
		}
#endif		
		cerr << PLUGIN_ERROR << e.str() << endl;
		if (e.strict) strict = true;
		// e.print();
	}
	/// quit on hard errors like accessing NULL pointers or strict type conversion
	/// errors IF the user sets the quitOnErrorFlag in the plugin.
	/// Otherwise most errors will go unnoticed apart from
	/// a messages in the terminal.
	/// It would be best if hosts could catch an exception instead
	/// and display something meaningful to the user.
	if (strict && m_quitOnErrorFlag) exit(EXIT_FAILURE);

	// this would disable all outputs even if some are valid
	// if (process) m_processFailure = true;
	
}

