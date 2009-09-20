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

#ifndef _PYTHON_WRAPPER_PLUGIN_H_
#define _PYTHON_WRAPPER_PLUGIN_H_

#define _CLASS_METHOD_ m_class << "::" << method
#define PLUGIN_ERROR "ERROR: In Vampy plugin [" << _CLASS_METHOD_ << "]" << endl << "Cause: "
#define DEBUG_NAME "[Vampy::call] " << _CLASS_METHOD_ << " "
#define DEAFULT_RETURN "Method [" << _CLASS_METHOD_ << "] is not implemented." << endl << "Returning default value: " << rValue
#define FLAG_VALUE "Flag: " << flagName << ": " << ((rValue==0)?"False":"True")

#include "vamp-sdk/Plugin.h"
#include <Python.h>
// #include <typeinfo>
// #include <stdarg.h>
#include "PyTypeInterface.h"

#include "Mutex.h"

using std::string;
using std::cerr;
using std::endl;

enum eProcessType {
	not_implemented,
	legacyProcess,
	numpyProcess
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
	int m_processType;
	PyObject *m_pyProcess;
	InputDomain m_inputDomain;
	PyTypeInterface m_ti;
	bool m_quitOnErrorFlag;
	bool m_debugFlag;

	void setProcessType();
	
	PyObject* numpyProcessCall(const float *const *inputBuffers, Vamp::RealTime timestamp);
	PyObject* legacyProcessCall(const float *const *inputBuffers, Vamp::RealTime timestamp);
	
	bool getBooleanFlag(char flagName[],bool) const;
/*
		Flags may be used to control the behaviour of the interface.
		Flags can be set in any Vampy plugin's __init__() function.
		Their scope is limited to an instance.
		Default values for all flags are False.
		Python Example:
		def __init__(self,inputSampleRate):
			self.use_strict_type_conversion = True
			self.vampy_debug_messages = True
			self.use_realtime_timestamp = False
			self.use_numpy_interface = False
			self.quit_on_type_error = False
			
*/
	
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
	
	template<typename RET> 
	RET &genericMethodCall(char *method, RET &rValue) const
	{
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if ( PyObject_HasAttrString(m_pyInstance,method) ) 
		{
			PyObject *pyValue = PyObject_CallMethod(m_pyInstance, method, NULL);
			if (pyValue) {
				m_ti.PyValue_To_rValue(pyValue,rValue);
				if (!m_ti.error) {
					Py_DECREF(pyValue);
					return rValue;
				} else {
					cerr << PLUGIN_ERROR << m_ti.lastError().message << endl;
					Py_CLEAR(pyValue);
					if (m_quitOnErrorFlag) exit(EXIT_FAILURE);
					return rValue;
				}
			} else {
				cerr << PLUGIN_ERROR << "Failed to call method." << endl;
				if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
				return rValue;
			}
		}
		// TODO: this fails to generalise because the << operator
		// doesn't accept all types.
		// if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
		return rValue;
	}

	template<typename RET,typename A1>
	RET genericMethodCallArgs(char *method, A1 arg1) const
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			// if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
			return rValue;
		}
		
		// These functions always return valid PyObjects 
		PyObject *pyMethod = m_ti.PyValue_From_CValue(method);
		PyObject* pyTuple = PyTuple_New(3);
		if (!pyTuple) return rValue;
		
		PyObject *pyArg1 = m_ti.PyValue_From_CValue(arg1);
				
		PyObject *pyValue = PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyArg1,NULL);
		if (!pyValue) 
		{
			cerr << PLUGIN_ERROR << "Failed to call method." << endl;
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		}
			
		Py_DECREF(pyMethod);
		Py_DECREF(pyArg1);

		m_ti.PyValue_To_rValue(pyValue,rValue);
		if (!m_ti.error) {
			Py_DECREF(pyValue);
		} else {
			cerr << PLUGIN_ERROR << m_ti.lastError().message << endl;
			Py_CLEAR(pyValue);
			if (m_quitOnErrorFlag) exit(EXIT_FAILURE);
		}
		return rValue;
	}

	template<typename RET,typename A1,typename A2>
	RET genericMethodCallArgs(char *method, A1 arg1, A2 arg2)
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			// if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
			return rValue;
		}
		
		// These functions always return valid PyObjects 
		PyObject *pyMethod = m_ti.PyValue_From_CValue(method);
		PyObject* pyTuple = PyTuple_New(3);
		if (!pyTuple) return rValue;
		
		PyObject *pyArg1 = m_ti.PyValue_From_CValue(arg1);
		PyObject *pyArg2 = m_ti.PyValue_From_CValue(arg2);
				
		PyObject *pyValue = PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyArg1,pyArg2,NULL);
		if (!pyValue) 
		{
			cerr << PLUGIN_ERROR << "Failed to call method." << endl;
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		}
			
		Py_DECREF(pyMethod);
		Py_DECREF(pyArg1);
		Py_DECREF(pyArg2);

		m_ti.PyValue_To_rValue(pyValue,rValue);
		if (!m_ti.error) {
			Py_DECREF(pyValue);
		} else {
			cerr << PLUGIN_ERROR << m_ti.lastError().message << endl;
			Py_CLEAR(pyValue);
			if (m_quitOnErrorFlag) exit(EXIT_FAILURE);
		}
		return rValue;
	}

	
	template<typename RET,typename A1,typename A2,typename A3>
	RET genericMethodCallArgs(char *method, A1 arg1, A2 arg2, A3 arg3)
	{
		RET rValue = RET();
		if (m_debugFlag) cerr << DEBUG_NAME << endl;
		if (!PyObject_HasAttrString(m_pyInstance,method)) {
			if (m_debugFlag) cerr << DEAFULT_RETURN << endl;
			return rValue;
		}
		
		// These functions always return valid PyObjects 
		PyObject *pyMethod = m_ti.PyValue_From_CValue(method);
		PyObject* pyTuple = PyTuple_New(3);
		if (!pyTuple) return rValue;
		
		PyObject *pyArg1 = m_ti.PyValue_From_CValue(arg1);
		PyObject *pyArg2 = m_ti.PyValue_From_CValue(arg2);
		PyObject *pyArg3 = m_ti.PyValue_From_CValue(arg3);
		
		// TODO: Pack it in a tuple to avoid va_list parsing!
		
		// callable = PyObject_GetAttr(callable, name);
		// if (callable == NULL)
		// 	return NULL;
		// PyObject* args; // pyTuple of input arguments
		//tmp = PyObject_Call(callable, args, NULL);
		
		
		PyObject *pyValue = PyObject_CallMethodObjArgs(m_pyInstance,pyMethod,pyArg1,pyArg2,pyArg3,NULL);
		if (!pyValue) 
		{
			cerr << PLUGIN_ERROR << "Failed to call method." << endl;
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		}
			
		Py_DECREF(pyMethod);
		Py_DECREF(pyArg1);
		Py_DECREF(pyArg2);
		Py_DECREF(pyArg3);

		m_ti.PyValue_To_rValue(pyValue,rValue);
		if (!m_ti.error) {
			Py_DECREF(pyValue);
		} else {
			cerr << PLUGIN_ERROR << m_ti.lastError().message << endl;
			Py_CLEAR(pyValue);
			if (m_quitOnErrorFlag) exit(EXIT_FAILURE);
		}
		return rValue;
	}

};


#endif
