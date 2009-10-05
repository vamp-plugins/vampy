/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

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

#ifndef _PYTHON_WRAPPER_PLUGIN_H_
#define _PYTHON_WRAPPER_PLUGIN_H_

#define _CLASS_METHOD_ m_class << "::" << method
#define PLUGIN_ERROR "ERROR: In Vampy plugin [" << _CLASS_METHOD_ << "]" << endl << "Cause: "
#define DEBUG_NAME "[Vampy::call] " << _CLASS_METHOD_ << " "
#define DEAFULT_RETURN "Method [" << _CLASS_METHOD_ << "] is not implemented. Returning default value."
#define FLAG_VALUE "Flag: " << flagName << ": " << ((rValue==0)?"False":"True")

#include <Python.h>
#include "PyExtensionModule.h"
#include "PyTypeInterface.h"
#include "vamp-sdk/Plugin.h"
#include "Mutex.h"

using std::string;
using std::cerr;
using std::endl;

enum eProcessType {
	not_implemented,
	legacyProcess,
	numpyProcess,
	numpy_bufferProcess,
	numpy_arrayProcess
	};

class PyPlugin : public Vamp::Plugin
{
public:
	PyPlugin(std::string plugin,float inputSampleRate, PyObject *pyClass, int &instcount);
	virtual ~PyPlugin();

	bool initialise(size_t channels, size_t stepSize, size_t blockSize);
	void reset();

	InputDomain getInputDomain() const;
	size_t getPreferredBlockSize() const;
	size_t getPreferredStepSize() const; 
	size_t getMinChannelCount() const; 
	size_t getMaxChannelCount() const;

	std::string getIdentifier() const;
	std::string getName() const;
	std::string getDescription() const;
	std::string getMaker() const;
	int getPluginVersion() const;
	std::string getCopyright() const;
	
	OutputList getOutputDescriptors() const;
	ParameterList getParameterDescriptors() const;
	float getParameter(std::string paramid) const;
	void setParameter(std::string paramid, float newval);
    
	FeatureSet process(const float *const *inputBuffers,
			   Vamp::RealTime timestamp);

	FeatureSet getRemainingFeatures();
	
protected:
	static Mutex m_pythonInterpreterMutex;
	PyObject *m_pyClass;
	PyObject *m_pyInstance;
	int &m_instcount;
	size_t m_stepSize;
	size_t m_blockSize;
	size_t m_channels;
	std::string m_plugin;
	std::string m_class;
	std::string m_path;
	eProcessType m_processType;
	PyObject *m_pyProcess;
	PyObject *m_pyProcessCallable;
	mutable InputDomain m_inputDomain;
	PyTypeInterface m_ti;
	int m_vampyFlags;
	bool m_quitOnErrorFlag;
	bool m_debugFlag;
	bool m_useRealTimeFlag;

	void setProcessType();
	
	FeatureSet processMethodCall(const float *const *inputBuffers,Vamp::RealTime timestamp);

	bool getBooleanFlag(char flagName[],bool) const;
	int getBinaryFlags(char flagName[], eVampyFlags) const;
	void typeErrorHandler(char *method) const;

	/// simple 'void return' call with no args
	void genericMethodCall(char *method) const
	{
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if ( PyObject_HasAttrString(m_pyInstance,method) ) 
		{
			PyObject *pyValue = PyObject_CallMethod(m_pyInstance, method, NULL);
			if (!pyValue) {
				cerr << PLUGIN_ERROR << "Failed to call method." << endl;
				if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			}
		}
	}

	/// 'no arg with default return value' call
	template<typename RET> 
	RET &genericMethodCall(char *method, RET &rValue) const
	{
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if ( PyObject_HasAttrString(m_pyInstance,method) ) 
		{
			PyObject *pyValue = PyObject_CallMethod(m_pyInstance, method, NULL);
			if (!pyValue) {
				cerr << PLUGIN_ERROR << "Failed to call method." << endl;
				if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
				return rValue;
			}

            /// convert the returned value
			m_ti.PyValue_To_rValue(pyValue,rValue);
			if (!m_ti.error) {
				Py_DECREF(pyValue);
			} else {
				Py_CLEAR(pyValue);
				typeErrorHandler(method);
			}
			return rValue;
		}
		if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
		return rValue;
	}

	/// unary call
	template<typename RET,typename A1>
	RET genericMethodCallArgs(char *method, A1 arg1) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
			return rValue;
		}
		
		/// prepare arguments for fast method call
		PyObject *pyMethod = m_ti.PyValue_From_CValue(method);
		PyObject *pyCallable = PyObject_GetAttr(m_pyInstance,pyMethod);
		PyObject* pyArgs = PyTuple_New(1);
		if (!(pyArgs && pyCallable && pyMethod)) {
			cerr << PLUGIN_ERROR << "Failed to prepare argument for calling method." << endl;
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyObject *pyArg1 = m_ti.PyValue_From_CValue(arg1);
		if (m_ti.error) {
			cerr << PLUGIN_ERROR << "Failed to convert argument for calling method." << endl;
			typeErrorHandler(method);
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArg1);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyTuple_SET_ITEM(pyArgs, 0, pyArg1);
		Py_INCREF(pyArg1);	
		
        /// call the method
		PyObject *pyValue = PyObject_Call(pyCallable,pyArgs,NULL);
		if (!pyValue) 
		{
			cerr << PLUGIN_ERROR << "Failed to call method." << endl;
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArg1);
			Py_CLEAR(pyArgs);
			return rValue;
		}
			
		Py_DECREF(pyMethod);
		Py_DECREF(pyCallable);
		Py_DECREF(pyArg1);
		Py_DECREF(pyArgs);    
		
		/// convert the returned value
		m_ti.PyValue_To_rValue(pyValue,rValue);
		if (!m_ti.error) {
			Py_DECREF(pyValue);
		} else {
			Py_CLEAR(pyValue);
			typeErrorHandler(method);
		}
		return rValue;
	}

	/// binary call
	template<typename RET,typename A1,typename A2>
	RET genericMethodCallArgs(char *method, A1 arg1, A2 arg2) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
			return rValue;
		}
		
		/// prepare arguments for fast method call
		PyObject *pyMethod = m_ti.PyValue_From_CValue(method);
		PyObject *pyCallable = PyObject_GetAttr(m_pyInstance,pyMethod);
		PyObject* pyArgs = PyTuple_New(2);
		if (!(pyArgs && pyCallable && pyMethod)) {
			cerr << PLUGIN_ERROR << "Failed to prepare arguments for calling method." << endl;
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyObject *pyArg1 = m_ti.PyValue_From_CValue(arg1);
		PyObject *pyArg2 = m_ti.PyValue_From_CValue(arg2);
		if (m_ti.error) {
			cerr << PLUGIN_ERROR << "Failed to convert arguments for calling method." << endl;
			typeErrorHandler(method);
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArg1);
			Py_CLEAR(pyArg2);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyTuple_SET_ITEM(pyArgs, 0, pyArg1);
		Py_INCREF(pyArg1);	
		PyTuple_SET_ITEM(pyArgs, 1, pyArg2);
		Py_INCREF(pyArg2);

		// calls the method
		PyObject *pyValue = PyObject_Call(pyCallable,pyArgs,NULL);
		if (!pyValue) 
		{
			cerr << PLUGIN_ERROR << "Failed to call method." << endl;
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArg1);
			Py_CLEAR(pyArg2);
			Py_CLEAR(pyArgs);
			return rValue;
		}
			
		Py_DECREF(pyMethod);
		Py_DECREF(pyCallable);
		Py_DECREF(pyArg1);
		Py_DECREF(pyArg2);
		Py_DECREF(pyArgs);    
		
		/// convert the returned value
		m_ti.PyValue_To_rValue(pyValue,rValue);
		if (!m_ti.error) {
			Py_DECREF(pyValue);
		} else {
			Py_CLEAR(pyValue);
			typeErrorHandler(method);
		}
		return rValue;
	}

	/// trenary call
	template<typename RET,typename A1,typename A2,typename A3>
	RET genericMethodCallArgs(char *method, A1 arg1, A2 arg2, A3 arg3) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
			return rValue;
		}
		
		/// prepare arguments for fast method call
		PyObject *pyMethod = m_ti.PyValue_From_CValue(method);
		PyObject *pyCallable = PyObject_GetAttr(m_pyInstance,pyMethod);
		PyObject* pyArgs = PyTuple_New(3);
		if (!(pyArgs && pyCallable && pyMethod)) {
			cerr << PLUGIN_ERROR << "Failed to prepare arguments for calling method." << endl;
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyObject *pyArg1 = m_ti.PyValue_From_CValue(arg1);
		PyObject *pyArg2 = m_ti.PyValue_From_CValue(arg2);
		PyObject *pyArg3 = m_ti.PyValue_From_CValue(arg3);
		if (m_ti.error) {
			cerr << PLUGIN_ERROR << "Failed to convert arguments for calling method." << endl;
			typeErrorHandler(method);
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArg1);
			Py_CLEAR(pyArg2);
			Py_CLEAR(pyArg3);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		/// Optimization: Pack args in a tuple to avoid va_list parsing.
		PyTuple_SET_ITEM(pyArgs, 0, pyArg1);
		Py_INCREF(pyArg1);	
		PyTuple_SET_ITEM(pyArgs, 1, pyArg2);
		Py_INCREF(pyArg2);
		PyTuple_SET_ITEM(pyArgs, 2, pyArg3);
		Py_INCREF(pyArg3);

		// PyObject *pyValue = PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyArg1,pyArg2,pyArg3,NULL);
		/// fast method call
		PyObject *pyValue = PyObject_Call(pyCallable,pyArgs,NULL);
		if (!pyValue) 
		{
			cerr << PLUGIN_ERROR << "Failed to call method." << endl;
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArg1);
			Py_CLEAR(pyArg2);
			Py_CLEAR(pyArg3);
			Py_CLEAR(pyArgs);
			return rValue;
		}
			
		Py_DECREF(pyMethod);
		Py_DECREF(pyCallable);
		Py_DECREF(pyArg1);
		Py_DECREF(pyArg2);
		Py_DECREF(pyArg3);    
		Py_DECREF(pyArgs);    
		
		/// convert the returned value
		m_ti.PyValue_To_rValue(pyValue,rValue);
		if (!m_ti.error) {
			Py_DECREF(pyValue);
		} else {
			Py_CLEAR(pyValue);
			typeErrorHandler(method);
		}
		return rValue;
	}

};

/// optimised process call
inline PyPlugin::FeatureSet
PyPlugin::processMethodCall(const float *const *inputBuffers,Vamp::RealTime timestamp)
{
	
	/// Optimizations: 1) we avoid ...ObjArg functions since we know
	/// the number of arguments, and we don't like va_list parsing 
	/// in the process. 2) Also: we're supposed to incref args, 
	/// but instead, we let the arguments tuple steal the references
	/// and decref them when it is deallocated.
	/// 3) all conversions are now using the fast sequence protocol
	/// (indexing the underlying object array).
	
	FeatureSet rFeatureSet;
	PyObject *pyChannelList = NULL;

	if (m_processType == numpy_bufferProcess) {
		pyChannelList = m_ti.InputBuffers_As_SharedMemoryList(inputBuffers,m_channels,m_blockSize);
	} 

	if (m_processType == legacyProcess) {
		pyChannelList = m_ti.InputBuffers_As_PythonLists(inputBuffers,m_channels,m_blockSize,m_inputDomain);
	}

#ifdef HAVE_NUMPY
	if (m_processType == numpy_arrayProcess) {
		pyChannelList = m_ti.InputBuffers_As_NumpyArray(inputBuffers,m_channels,m_blockSize,m_inputDomain);
	}
#endif

/// we don't expect these to fail unless out of memory (which is very unlikely on modern systems)
#ifdef _DEBUG
	if (!pyChannelList) {
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		std::string method = PyString_AsString(m_pyProcess);
		cerr << PLUGIN_ERROR << "Failed to create channel list." << endl;
		return rFeatureSet;
	}
#endif		

	PyObject *pyTimeStamp = NULL;
		
	if (m_useRealTimeFlag) {
		//(1) pass TimeStamp as PyRealTime object
		pyTimeStamp = PyRealTime_FromRealTime(timestamp);

	} else {
		//(2) pass TimeStamp as frame count (long Sample Count)
		pyTimeStamp = PyLong_FromLong(Vamp::RealTime::realTime2Frame 
		(timestamp, (unsigned int) m_inputSampleRate));
	}


#ifdef _DEBUG
	if (!pyTimeStamp) {
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		std::string method = PyString_AsString(m_pyProcess);
		cerr << PLUGIN_ERROR << "Failed to create RealTime time stamp." << endl;
		Py_DECREF(pyChannelList);
		return rFeatureSet;
	}
#endif

	/// Old method: Call python process (returns new reference)
	/// PyObject *pyValue = PyObject_CallMethodObjArgs
	/// (m_pyInstance,m_pyProcess,pyChannelList,pyTimeStamp,NULL);
	
	PyObject *pyArgs = PyTuple_New(2);
	PyTuple_SET_ITEM(pyArgs, 0, pyChannelList); 
	PyTuple_SET_ITEM(pyArgs, 1, pyTimeStamp); 

	/// Call python process (returns new reference) {kwArgs = NULL}
	PyObject *pyValue = PyObject_Call(m_pyProcessCallable,pyArgs,NULL);

	if (!pyValue) {
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		std::string method = PyString_AsString(m_pyProcess);
		cerr << PLUGIN_ERROR << "An error occurred while evaluating Python process." << endl;
		Py_CLEAR(pyValue);
		Py_CLEAR(pyArgs);
		return rFeatureSet;
	}
        
	rFeatureSet = m_ti.PyValue_To_FeatureSet(pyValue);
	if (!m_ti.error) {
		Py_DECREF(pyValue);
		Py_DECREF(pyArgs);
	} else {
		typeErrorHandler(PyString_AsString(m_pyProcess));
		Py_CLEAR(pyValue);
		Py_CLEAR(pyArgs);
	}
	return rFeatureSet;
}

#endif
