/**
 * This Vamp plugin is a wrapper for Python Scripts. (VamPy)
 * Centre for Digital Music, Queen Mary, University of London.
 * Copyright 2008, George Fazekas.

*/


#include "PyPlugScanner.h"

//#include <fstream>
//#include <cctype>

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#define pathsep ("\\")
#else 
#include <dirent.h>
#include <dlfcn.h>
#define pathsep ("/")
#endif 
#define joinPath(a,b) ( (a)+pathsep+(b) )

using std::string;
using std::vector;
using std::cerr;
using std::endl;

PyPlugScanner::PyPlugScanner()
{

} 

PyPlugScanner *PyPlugScanner::m_instance = NULL;
bool PyPlugScanner::m_hasInstance = false;

PyPlugScanner*
PyPlugScanner::getInstance()
{
	if (!m_hasInstance) {
		m_instance = new PyPlugScanner();
		m_hasInstance = true;
	}
	return m_instance;
}

void
PyPlugScanner::setPath(vector<string> path)
{
	m_path=path;
}

// TODO: This should return all scripts for all valid paths
// Validate python classes here?
// For now, we assume that each script on the path has one valid class
vector<string> 
PyPlugScanner::getPyPlugs()
{
	//foreach m_path listFiles then return vector<pyPlugs>
	//format: FullPathString/FileName.py:ClassName
	
	vector<string> pyPlugs;
	string pluginKey;
	PyObject *pyClass;
	
    for (size_t i = 0; i < m_path.size(); ++i) {
        
        vector<string> files = listFiles(m_path[i],"py");

        for (vector<string>::iterator fi = files.begin();
             fi != files.end(); ++fi) {
				string script = *fi;
				if (!script.empty()) {					
					string classname=script.substr(0,script.rfind('.'));
					pluginKey=joinPath(m_path[i],script)+":"+classname;
					pyClass = getScriptClass(m_path[i],classname);
					if (pyClass == NULL) 
					cerr << "Warning: Syntax error in VamPy plugin:  " 
					     << classname << ". Avoiding plugin." << endl;
					else { 
							pyPlugs.push_back(pluginKey);
							m_pyClasses.push_back(pyClass);
						}
					//pyPlugs.push_back(pluginKey);
				}
		}		
	}
		
return pyPlugs;	

}


//For now return one class object found in each script
vector<PyObject*> 
PyPlugScanner::getPyClasses()
{
return m_pyClasses;	

}


//Validate
//This should not be called more than once!
PyObject* 
PyPlugScanner::getScriptClass(string path, string classname)
{

	//Add plugin path to active Python Path 
	string pyCmd = "import sys\nsys.path.append('" + path + "')\n";
	PyRun_SimpleString(pyCmd.c_str());

	//Assign an object to the source code
	PyObject *pySource = PyString_FromString(classname.c_str());

	//Import it as a module into the py interpreter
	PyObject *pyModule = PyImport_Import(pySource);
	PyObject* pyError = PyErr_Occurred();
	if (! pyError == 0) { 
		cerr << "ERROR: error importing source: " << classname << endl; 
		PyErr_Print(); 
		Py_DECREF(pySource); 
		Py_CLEAR(pyModule);  // safer if pyModule==NULL
		return NULL; 
	}
	Py_DECREF(pySource);

	//Read the namespace of the module into a dictionary object (borrowed reference)
	PyObject *pyDict = PyModule_GetDict(pyModule);
	Py_DECREF(pyModule);

	//Get the PluginClass from the module (borrowed reference)
	PyObject *pyClass = PyDict_GetItemString(pyDict, classname.c_str());

	//Check if class is present and a callable method is implemented
	if (pyClass && PyCallable_Check(pyClass)) {

	    return pyClass;
	}	
	else {
		cerr << "ERROR: callable plugin class could not be found in source: " << classname << endl 
			<< "Hint: plugin source filename and plugin class name must be the same." << endl;
		PyErr_Print(); 
		return NULL;
	}
}



// Return a list of files in dir with given extension
// Code taken from hostext/PluginLoader.cpp
vector<string>
PyPlugScanner::listFiles(string dir, string extension)
{
    vector<string> files;

#ifdef _WIN32

    string expression = dir + "\\*." + extension;
    WIN32_FIND_DATA data;
    HANDLE fh = FindFirstFile(expression.c_str(), &data);
    if (fh == INVALID_HANDLE_VALUE) return files;

    bool ok = true;
    while (ok) {
        files.push_back(data.cFileName);
        ok = FindNextFile(fh, &data);
    }

    FindClose(fh);

#else

    size_t extlen = extension.length();
    DIR *d = opendir(dir.c_str());
    if (!d) return files;
            
    struct dirent *e = 0;
    while ((e = readdir(d))) {
 
        if (!(e->d_type & DT_REG) && (e->d_type != DT_UNKNOWN)) continue;
        
        if (!e->d_name) continue;
       
        size_t len = strlen(e->d_name);
        if (len < extlen + 2 ||
            e->d_name + len - extlen - 1 != "." + extension) {
            continue;
        }
		//cerr << "pyscripts: " << e->d_name <<  endl;
        files.push_back(e->d_name);
    }

    closedir(d);
#endif

    return files;
}

//Return correct plugin directories as per platform
//Code taken from vamp-sdk/PluginHostAdapter.cpp

//!!! It would probably be better to actually call
// PluginHostAdapter::getPluginPath.  That would mean this "plugin"
// needs to link against vamp-hostsdk, but that's probably acceptable
// as it is sort of a host as well.

std::vector<std::string>
PyPlugScanner::getAllValidPath()
{
    std::vector<std::string> path;
    std::string envPath;

    char *cpath = getenv("VAMP_PATH");
    if (cpath) envPath = cpath;

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#define DEFAULT_VAMP_PATH "%ProgramFiles%\\Vamp Plugins"
#else
#define PATH_SEPARATOR ':'
#ifdef __APPLE__
#define DEFAULT_VAMP_PATH "$HOME/Library/Audio/Plug-Ins/Vamp:/Library/Audio/Plug-Ins/Vamp"
#else
#define DEFAULT_VAMP_PATH "$HOME/vamp:$HOME/.vamp:/usr/local/lib/vamp:/usr/lib/vamp"
#endif
#endif

    if (envPath == "") {
        envPath = DEFAULT_VAMP_PATH;
        char *chome = getenv("HOME");
        if (chome) {
            std::string home(chome);
            std::string::size_type f;
            while ((f = envPath.find("$HOME")) != std::string::npos &&
                    f < envPath.length()) {
                envPath.replace(f, 5, home);
            }
        }
#ifdef _WIN32
        char *cpfiles = getenv("ProgramFiles");
        if (!cpfiles) cpfiles = "C:\\Program Files";
        std::string pfiles(cpfiles);
        std::string::size_type f;
        while ((f = envPath.find("%ProgramFiles%")) != std::string::npos &&
               f < envPath.length()) {
            envPath.replace(f, 14, pfiles);
        }
#endif
    }

    std::string::size_type index = 0, newindex = 0;

    while ((newindex = envPath.find(PATH_SEPARATOR, index)) < envPath.size()) {
	path.push_back(envPath.substr(index, newindex - index));
	index = newindex + 1;
    }
    
    path.push_back(envPath.substr(index));

    return path;
}
