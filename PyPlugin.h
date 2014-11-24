/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#ifndef _PYTHON_WRAPPER_PLUGIN_H_
#define _PYTHON_WRAPPER_PLUGIN_H_

#define _CLASS_METHOD_ m_class << "::" << method
#define PLUGIN_ERROR "ERROR: In Vampy plugin [" << _CLASS_METHOD_ << "]" << endl << "Cause: "
#define DEBUG_NAME "[Vampy::call] " << _CLASS_METHOD_ << " "
#define DEFAULT_RETURN "Method [" << _CLASS_METHOD_ << "] is not implemented. Returning default value."
#define FLAG_VALUE "Flag: " << flagName << ": " << ((rValue==0)?"False":"True")

#include <Python.h>
#include "PyExtensionModule.h"
#include "PyTypeInterface.h"
#include "PyTypeConversions.h"
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
	PyPlugin(std::string plugin,float inputSampleRate, PyObject *pyClass, int &instcount, bool &numpyInstalled);
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
	PyTypeConversions m_tc;
	PyTypeInterface m_ti;
	int m_vampyFlags;
	bool m_quitOnErrorFlag;
	bool m_debugFlag;
	bool m_useRealTimeFlag;
	bool m_numpyInstalled;
	mutable bool m_processFailure;

	void setProcessType();
	
	FeatureSet processMethodCall(const float *const *inputBuffers,Vamp::RealTime timestamp);

	bool getBooleanFlag(const char flagName[],bool) const;
	int getBinaryFlags(const char flagName[], eVampyFlags) const;
	void typeErrorHandler(const char *method, bool process = false) const;

	/// simple 'void return' call with no args
	void genericMethodCall(const char *method) const
	{
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if ( PyObject_HasAttrString(m_pyInstance,method) ) 
		{
		    PyObject *pyValue = PyObject_CallMethod(m_pyInstance, (char *)method, NULL);
			if (!pyValue) {
				cerr << PLUGIN_ERROR << "Failed to call method." << endl;
				if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			}
		}
	}

	/// 'no arg with default return value' call
	template<typename RET> 
	RET &genericMethodCall(const char *method, RET &rValue) const
	{
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if ( PyObject_HasAttrString(m_pyInstance,method) ) 
		{
		    PyObject *pyValue = PyObject_CallMethod(m_pyInstance, (char *)method, NULL);
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
		if (m_debugFlag) cerr << DEFAULT_RETURN << endl;
		return rValue;
	}

	/// unary call
	template<typename RET,typename A1>
	RET genericMethodCallArgs(const char *method, A1 arg1) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEFAULT_RETURN << endl;
			return rValue;
		}
		
		/// prepare arguments for fast method call
		PyObject *pyMethod = m_tc.PyValue_From_CValue(method);
		PyObject *pyCallable = PyObject_GetAttr(m_pyInstance,pyMethod);
		PyObject* pyArgs = PyTuple_New(1);
		if (!(pyArgs && pyCallable && pyMethod)) {
			cerr << PLUGIN_ERROR << "Failed to prepare argument for calling method." << endl;
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyObject *pyArg1 = m_tc.PyValue_From_CValue(arg1);
		if (m_tc.error) {
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
	RET genericMethodCallArgs(const char *method, A1 arg1, A2 arg2) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEFAULT_RETURN << endl;
			return rValue;
		}
		
		/// prepare arguments for fast method call
		PyObject *pyMethod = m_tc.PyValue_From_CValue(method);
		PyObject *pyCallable = PyObject_GetAttr(m_pyInstance,pyMethod);
		PyObject* pyArgs = PyTuple_New(2);
		if (!(pyArgs && pyCallable && pyMethod)) {
			cerr << PLUGIN_ERROR << "Failed to prepare arguments for calling method." << endl;
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyObject *pyArg1 = m_tc.PyValue_From_CValue(arg1);
		PyObject *pyArg2 = m_tc.PyValue_From_CValue(arg2);
		if (m_tc.error) {
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
	RET genericMethodCallArgs(const char *method, A1 arg1, A2 arg2, A3 arg3) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEFAULT_RETURN << endl;
			return rValue;
		}
		
		/// prepare arguments for fast method call
		PyObject *pyMethod = m_tc.PyValue_From_CValue(method);
		PyObject *pyCallable = PyObject_GetAttr(m_pyInstance,pyMethod);
		PyObject* pyArgs = PyTuple_New(3);
		if (!(pyArgs && pyCallable && pyMethod)) {
			cerr << PLUGIN_ERROR << "Failed to prepare arguments for calling method." << endl;
			Py_CLEAR(pyMethod);
			Py_CLEAR(pyCallable);
			Py_CLEAR(pyArgs);
			return rValue;
		}
		
		PyObject *pyArg1 = m_tc.PyValue_From_CValue(arg1);
		PyObject *pyArg2 = m_tc.PyValue_From_CValue(arg2);
		PyObject *pyArg3 = m_tc.PyValue_From_CValue(arg3);
		if (m_tc.error) {
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

#endif
