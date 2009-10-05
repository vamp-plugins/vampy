/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

/*
PyExtensionManager: This class is responsible for initialisation
and cleanup of the extension module, as well as the loaded plugin
module namespaces. 

NOTES: Why do we need to clean up the module?

The module exposed by Vampy to the embedded interpreter contains 
callback functions. These functions are accessed via function 
pointers stored in the extension module's namespace dictionary.

Unfortunately, when the shared library is unloaded and reloaded 
during a host session, these addresses might change. 
Therefore, we reinitialise the module dict before each use. 
However, this will cause garbage collection errors or segmentation 
faults, when elements of the dict of the previous session are 
attempted to free. Therefore, we clear the dictinary describing
the module namespace and replace all fuction pointers with Py_None
objects in individual plugin module namespaces. The reference
count on these can be safely decremented next time vampy is loaded
and the namespaces are reinitialised.

Why doesn't the GC clean this up correctly?

In a normal Python session the GC would deallocate the module
dict at the end. In embedded python, although the GC appears
to be called when the shared lib is unloaded, the interpreter
is reused. Since there is no C/API call to unload modules,
and at the time of unloading vampy the wrapped function pointers
are still valid, the GC doesn't collect them, nor are they freed
by the interpreter. When vampy is reloaded however, the module
dict will contain invalid addresses. The above procedure solves
this problem.


*/


#ifndef _PYEXTENSIONMANAGER_H_
#define _PYEXTENSIONMANAGER_H_

using std::cerr;
using std::endl;
using std::string;
using std::vector;

class PyExtensionManager
{
public:
	PyExtensionManager();
	~PyExtensionManager();
	bool initExtension();
	void setPlugModuleNames(vector<string> pyPlugs);
	void deleteModuleName(string plugKey);

private:
	static char* m_exposedNames[];
	
	vector<string> m_plugModuleNames;
	PyObject* m_pyGlobalNamespace;
	PyObject* m_pyVampyNamespace;

	void cleanAllLocals() const;
	void cleanLocalNamespace(const char* plugModuleName) const;
	void updateAllLocals() const;
	void updateLocalNamespace(const char* plugModuleName) const;

	void printDict(PyObject* inDict) const;
	bool cleanModule() const;

};

#endif


