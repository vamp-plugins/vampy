/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/


#include "PyPlugScanner.h"
#include <algorithm>
#include <cstdlib>
//#include "vamp-hostsdk/PluginHostAdapter.h"

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
using std::find;

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

// We assume that each script on the path has one valid class
vector<string> 
PyPlugScanner::getPyPlugs()
{
	//for_each m_path listFiles then return vector<pyPlugs>
	//key format: FullPathString/FileName.py:ClassName
	
	bool getCompiled = true;
	char* getPyc = getenv("VAMPY_COMPILED");
	if (getPyc) {
		string value(getPyc);
		cerr << "VAMPY_COMPILED=" << value << endl;
		getCompiled = value.compare("1")?false:true;
	}
	
	vector<string> pyPlugs;
	string pluginKey;
	PyObject *pyClass;
	
    for (size_t i = 0; i < m_path.size(); ++i) {
        
        vector<string> files = listFiles(m_path[i],"py");

        /// recognise byte compiled plugins
		if (getCompiled) {
        	vector<string> pyc_files = listFiles(m_path[i],"pyc");
        	vector<string> pyo_files = listFiles(m_path[i],"pyo");
			mergeFileLists(pyc_files,pyo_files,".pyo");
			mergeFileLists(pyo_files,files,".py");
		}

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

/// insert python byte code names (.pyc) if a .py file can not be found
/// The interpreter automatically generates byte code files and executes
/// them if they exist. Therefore, we prefer .py files, but we allow
/// (relatively) closed source distributions by recognising .pyc files.
void
PyPlugScanner::mergeFileLists(vector<string> &src, vector<string> &tg, string target_ext)
{
    for (vector<string>::iterator srcit = src.begin();
    	srcit != src.end(); ++srcit) {
			// cerr << *srcit;
			string src_name = *srcit;
			string tg_name = src_name.substr(0,src_name.rfind('.')) + target_ext;
			vector<string>::iterator tgit = find (tg.begin(), tg.end(), tg_name);
			if (tgit == tg.end()) tg.push_back(src_name);
	}
	
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
	if (pyError) { 
		cerr << "ERROR: error importing source: " << classname << endl; 
		PyErr_Print(); 
		Py_DECREF(pySource); 
		Py_CLEAR(pyModule);  // safer if pyModule==NULL
		return NULL; 
	}
	Py_DECREF(pySource);

	//Read the dictionary object holding the namespace of the module (borrowed reference)
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


//!!! It would probably be better to actually call
// PluginHostAdapter::getPluginPath.  That would mean this "plugin"
// needs to link against vamp-hostsdk, but that's probably acceptable
// as it is sort of a host as well.

// std::vector<std::string>
// PyPlugScanner::getAllValidPath()
// { 
// 	Vamp::PluginHostAdapter host_adapter( ??? );
// 	return host_adapter.getPluginPath();
// }

// tried to implement it, but found a bit confusing how to 
// instantiate the host adapter here...


//Return correct plugin directories as per platform
//Code taken from vamp-sdk/PluginHostAdapter.cpp
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

	//can add an extra path for vampy plugins
	char* extraPath = getenv("VAMPY_EXTPATH");
	if (extraPath) {
		string vampyPath(extraPath);
		cerr << "VAMPY_EXTPATH=" << vampyPath << endl;
		path.push_back(vampyPath);
	}
	
    return path;
}
