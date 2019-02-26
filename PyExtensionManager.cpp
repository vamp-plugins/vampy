/* -*- c-basic-offset: 8 indent-tabs-mode: t -*- */
/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>
#include "vamp/vamp.h"
#include "PyExtensionModule.h"
#include "PyExtensionManager.h"
#include <algorithm>
#include "Debug.h"

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::find;

//static
const char* PyExtensionManager::m_exposedNames[] = {
	
		"ParameterDescriptor",
		"OutputDescriptor",
		"FeatureSet",
		"Feature",
		"RealTime",
		"frame2RealTime",

/* 		// using builtin objects:
		"ParameterList",
		"OutputList",
		"FeatureList",
		"OneSamplePerStep",
		"FixedSampleRate",
		"VariableSampleRate",
		"TimeDomain",
		"FrequencyDomain",
*/
		
		NULL
};

PyExtensionManager::PyExtensionManager() :
	m_pyGlobalNamespace(0),
	m_pyVampyNamespace(0)
{
	DSTREAM << "Creating extension manager." << endl;
}

bool 
PyExtensionManager::initExtension() 
{
	DSTREAM << "Initialising extension module." << endl; 

	/// call the module initialiser first
	initvampy();

	/// these references are all borrowed
	m_pyGlobalNamespace = PyImport_GetModuleDict();
	if (!m_pyGlobalNamespace) 
		{cerr << "Vampy::PyExtensionManager::initExtension: GlobalNamespace failed." << endl; return false;}
	PyObject *pyVampyModule = PyDict_GetItemString(m_pyGlobalNamespace,"vampy");
	if (!pyVampyModule) 
		{cerr << "Vampy::PyExtensionManager::initExtension: VampyModule failed." << endl; return false;}
	m_pyVampyNamespace = PyModule_GetDict(pyVampyModule);
	if (!m_pyVampyNamespace) 
		{cerr << "Vampy::PyExtensionManager::initExtension: VampyNamespace failed." << endl; return false;}
    
	/// initialise local namespaces
	updateAllLocals();

	DSTREAM << "Vampy: Extension namespaces updated." << endl; 

	return true;
}


PyExtensionManager::~PyExtensionManager()
{
	if (!m_pyVampyNamespace) {
		DSTREAM << "Vampy::~PyExtensionManager: manager was never initialised, or initialisation did not complete: not attempting cleanup" << endl;
		return;
	}
		
	DSTREAM << "Cleaning locals..." << endl;

	cleanAllLocals(); 

	DSTREAM << "Cleaning module..." << endl;

	if (!cleanModule()) {
		cerr << "Vampy::~PyExtensionManager: failed to clean extension module." << endl;
	} else {
		DSTREAM << "Vampy::~PyExtensionManager: Extension module cleaned." << endl;
	}
}



void
PyExtensionManager::setPlugModuleNames(vector<string> pyPlugs)
{
	for (size_t i = 0; i < pyPlugs.size(); ++i) {
		string modName = pyPlugs[i];
		string tmp = modName.substr(modName.rfind(':')+1,modName.size()-1);
		m_plugModuleNames.push_back(tmp);

		DSTREAM << "Inserted module name: " << tmp << endl;
	}
}

void
PyExtensionManager::deleteModuleName(string plugKey)
{
	string name = plugKey.substr(plugKey.rfind(':')+1,plugKey.size()-1);
	vector<string>::iterator it = 
		find (m_plugModuleNames.begin(), m_plugModuleNames.end(), name);
	if (it != m_plugModuleNames.end()) m_plugModuleNames.erase(it);

	DSTREAM << "PyExtensionManager::deleteModuleName: Deleted module name: " << name << endl;
}


void 
PyExtensionManager::cleanAllLocals() const
{
	for (size_t i = 0; i < m_plugModuleNames.size(); ++i) {
	    cleanLocalNamespace(m_plugModuleNames[i].c_str());
	}
}

void 
PyExtensionManager::updateAllLocals() const
{
	for (size_t i = 0; i < m_plugModuleNames.size(); ++i) {
		updateLocalNamespace(m_plugModuleNames[i].c_str());
	}
}

void 
PyExtensionManager::cleanLocalNamespace(const char* plugModuleName) const
{
	DSTREAM << "Cleaning local namespace: " << plugModuleName << endl;

	/// these references are all borrowed
	PyObject *pyPlugModule = PyDict_GetItemString(m_pyGlobalNamespace,plugModuleName);
	if (!pyPlugModule) return;
	PyObject *pyPlugDict = PyModule_GetDict(pyPlugModule);
	if (!pyPlugDict) return;
	
	int i = 0;
	while (PyExtensionManager::m_exposedNames[i]) {
		const char* name = PyExtensionManager::m_exposedNames[i];
		i++;
		PyObject *key = PyString_FromString(name);
		if (!key) break;
		if (PyDict_Contains(pyPlugDict,key)) {
			if (PyDict_SetItem(pyPlugDict,key,Py_None) != 0)
				cerr << "Vampy::PyExtensionManager::cleanLocalNamespace: Failed: " 
				<< name << " of "<< plugModuleName << endl;
			else DSTREAM << "Cleaned local name: " << name << endl;
		}
		Py_DECREF(key);
	}
}

void 
PyExtensionManager::updateLocalNamespace(const char* plugModuleName) const
{
	DSTREAM << "Updating local namespace: " << plugModuleName << endl;

	/// this allows the use of common syntax like:
	/// from vampy import * 
	/// even after several unload/reload cycles
		
	/// these references are all borrowed
	PyObject *pyPlugModule = PyDict_GetItemString(m_pyGlobalNamespace,plugModuleName);
	if (!pyPlugModule) return;
	PyObject *pyPlugDict = PyModule_GetDict(pyPlugModule);
	if (!pyPlugDict) return;
	
	int i = 0;
	while (PyExtensionManager::m_exposedNames[i]) {
		const char* name = PyExtensionManager::m_exposedNames[i];
		i++;
		PyObject *key = PyString_FromString(name);
		if (!key) break;
		if (PyDict_Contains(pyPlugDict,key)) {
			PyObject* item = PyDict_GetItem(m_pyVampyNamespace,key);
			if (PyDict_SetItem(pyPlugDict,key,item) != 0)
				cerr << "Vampy::PyExtensionManager::updateLocalNamespace: Failed: " 
				<< name << " of "<< plugModuleName << endl;
			else DSTREAM << "Updated local name: " << name << endl;
		} else {
			DSTREAM << "Local namespace does not contain name: " << name << endl;
		}
		Py_DECREF(key);
	}
}


bool 
PyExtensionManager::cleanModule(void) const
{
	PyObject *m = PyImport_AddModule("vampy");
	if (!m) {
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		cerr << "Vampy::PyExtensionManager::cleanModule: PyImport_AddModule returned NULL!" << endl;
		return false;
	} else {
		PyObject *dict = PyModule_GetDict(m);
#ifdef _DEBUG		
		Py_ssize_t ln = PyDict_Size(dict);
		DSTREAM << "Vampy::PyExtensionManager::cleanModule: Size of module dict = " << (int) ln << endl;
#endif		
		/// Clean the module dictionary.
		// printDict(dict);
		PyDict_Clear(dict);
		if (PyErr_Occurred()) 
			{ PyErr_Print(); PyErr_Clear(); return false; }
		PyObject *name = PyString_FromString("vampy");
		if (name) PyDict_SetItemString(dict,"__name__",name);
		Py_XDECREF(name);
#ifdef _DEBUG		
	    ln = PyDict_Size(dict);
		DSTREAM << "Vampy::PyExtensionManager::cleanModule: Size of module dict (cleaned) = " << (int) ln << endl;
#endif
		return true;
	}
}

void 
PyExtensionManager::printDict(PyObject* inDict) const
{
	Py_ssize_t pyPos = 0;
	PyObject *pyKey, *pyDictValue;
	cerr << endl << endl << "Module dictionary contents: " << endl;
	while (PyDict_Next(inDict, &pyPos, &pyKey, &pyDictValue))
	{ 
		char *key = PyString_AS_STRING(pyKey);
		char *val = PyString_AS_STRING(PyObject_Str(pyDictValue));
		cerr << "key: [ '" << key << "' ] value: " << val << endl;
	}
}

