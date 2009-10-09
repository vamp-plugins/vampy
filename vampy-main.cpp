/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>

#ifdef HAVE_NUMPY

// define a unique API pointer 
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API 
#include "numpy/arrayobject.h"

// prevent building with very old versions of numpy
#ifndef NPY_VERSION 
#undef HAVE_NUMPY
#endif

#endif

// this is not part of the API, but we will require it for a bug workaround
// define this symbol if you use another version of numpy in the makefile
// Vampy will not attempt to load a lower version than specified
#ifdef HAVE_NUMPY
#ifndef NUMPY_SHORTVERSION
#define NUMPY_SHORTVERSION 1.1 
#endif
#endif

#include "vamp/vamp.h"
#include "vamp-sdk/PluginAdapter.h"
#include "PyPlugScanner.h"
#include "PyPlugin.h"
#include "PyExtensionModule.h"
#include "PyExtensionManager.h"


#ifdef _WIN32
#define pathsep ('\\')
#include <windows.h>
#include <tchar.h>
#else 
#define pathsep ('/')
#include <dirent.h>
#include <dlfcn.h>
#endif

using std::cerr;
using std::endl;
using std::string;
using std::vector;

static int adinstcount;
static int totinstcount;
static bool numpyInstalled = false;
static bool arrayApiInitialised = false;

class PyPluginAdapter : public Vamp::PluginAdapterBase
{
public: 
    PyPluginAdapter(std::string pyPlugId, PyObject* pyClass) :
        PluginAdapterBase(),
        m_plug(pyPlugId),		
        m_pyClass(pyClass),
		m_failed(false)
    { 
        cerr << "PyPluginAdapter:ctor:"<< adinstcount << ": " << m_plug << endl; 
        adinstcount++;
    }
    
    ~PyPluginAdapter() 
    {
    }

	bool failed() { return m_failed; }
	std::string getPlugKey() { return m_plug; }

protected:
    Vamp::Plugin *createPlugin(float inputSampleRate)
    {
        try {
            PyPlugin *plugin = new PyPlugin(m_plug, inputSampleRate, m_pyClass, totinstcount, numpyInstalled);
            return plugin;
        } catch (...) {
            cerr << "PyPluginAdapter::createPlugin: Failed to construct PyPlugin" << endl;
			// any plugin with syntax errors will fail to construct
			m_failed = true;
            return 0;
        }
    }
    
    std::string m_plug;
    PyObject *m_pyClass;
	bool m_failed;  
};


static void array_API_initialiser()
{
	if (arrayApiInitialised) return; 

/* Numpy 1.3 build note: there seems to be a bug 
in this version (at least on OS/X) which will cause memory 
access error in the array API import function if an earlier runtime 
version of Numpy is used when loading the library.
(below is a horrible workaround)
*/

#ifdef HAVE_NUMPY

	string ver;
	float numpyVersion;

	/// attmept to test numpy version before importing the array API
	cerr << "Numpy build information: ABI level: " << NPY_VERSION 
	<< " Numpy version: " << NUMPY_SHORTVERSION << endl;
	
	PyObject *pyModule, *pyDict, *pyVer;
	
	pyModule = PyImport_ImportModule("numpy"); //numpy.core.multiarray
	if (!pyModule) {
		cerr << "Vampy was compiled with Numpy support but Numpy does not seem to be installed." << endl;
#ifdef __APPLE__
		cerr << "Hint: Check if Numpy is installed for the particular setup of Python used by Vampy (given by Python exec prefix)." << endl;
#endif		
		goto numpyFailure;
	}

	pyDict = PyModule_GetDict(pyModule); // borrowed ref
	if (!pyDict) {
		cerr << "Can not access Numpy module dictionary." << endl;
		goto numpyFailure;
	}

	pyVer = PyDict_GetItemString(pyDict,"__version__"); //borrowed ref
	if (!pyVer) {
		cerr << "Can not access Numpy version information." << endl;
		goto numpyFailure;
	}

	ver = PyString_AsString(pyVer);
	ver = ver.substr(0,ver.rfind("."));
	if(EOF == sscanf(ver.c_str(), "%f", &numpyVersion))
	{
		cerr << "Could not parse Numpy version information." << endl;
		goto numpyFailure;
	}

	cerr << "Numpy runtime version: " << numpyVersion << endl;
	if (numpyVersion < (float) NUMPY_SHORTVERSION) {
		cerr << "Incompatible Numpy version found: " << numpyVersion << endl;
		goto numpyFailure;
	}

	Py_DECREF(pyModule);

	// At least we catch import errors, but if binary compatibility
	// has changed without notice, this would still fail.
	// However, we should never get to this point now anyway.
	import_array();
	if (PyErr_Occurred()) { 
		cerr << "Import error while loading the Numpy Array API." << endl;
		PyErr_Print(); PyErr_Clear(); 
		goto numpyFailure;
	}
	else {

#ifdef _DEBUG		
		if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {  
			// the Import function does this check already.
			cerr << "Warning: Numpy version mismatch. (Build version: " 
				<< NPY_VERSION << " Runtime version: " << PyArray_GetNDArrayCVersion() << ")" << endl;
			goto numpyFailure; 
		}
#endif

		numpyInstalled = true;
		arrayApiInitialised = true;
		return;
  	}


numpyFailure: 
	cerr << "Please make sure you have Numpy " << NUMPY_SHORTVERSION << " or greater installed." << endl;
	cerr << "Vampy: Numpy support disabled." << endl;
	numpyInstalled = false;
	arrayApiInitialised = true;
	if (pyModule) Py_XDECREF(pyModule);
	return;

/*HAVE_NUMPY*/
#endif 

    numpyInstalled = false;
	arrayApiInitialised = true;
	return;
}


static std::vector<PyPluginAdapter *> adapters;
static bool haveScannedPlugins = false;

static bool tryPreload(string name)
{
//    cerr << "tryPreload: " << name << endl;
#ifdef _WIN32
    void *lib = LoadLibrary(name.c_str());
    if (!lib) {
        return false;
    }
#else
    void *lib = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        perror("dlopen");
        return false;
    }
#endif
    cerr << "Preloaded Python from " << name << endl;
    return true;
}

static bool preloadPython()
{
#ifdef _WIN32
    // this doesn't seem to be necessary at all on Windows
    return true;
#endif

    string pyver = Py_GetVersion();
    int dots = 2;
    string shortver;
    for (size_t i = 0; i < pyver.length(); ++i) {
        if (pyver[i] == '.') {
            if (--dots == 0) {
                shortver = pyver.substr(0, i);
                break;
            }
        }
    }
    cerr << "Short version: " << shortver << endl;
	// this is useful to find out where the loaded library might be loaded from
	cerr << "Python exec prefix: " << Py_GetExecPrefix() << endl;

    char *pylib = getenv("VAMPY_PYLIB");
    if (pylib && *pylib) {
        cerr << "Trying to preload Python from specified location " << pylib
	    << "..." << endl;
        return tryPreload(string(pylib));
    } 

    vector<string> pfxs;
    pfxs.push_back(string(Py_GetExecPrefix()) + "/");
    pfxs.push_back(string(Py_GetExecPrefix()) + "/lib/");
    pfxs.push_back("");
    pfxs.push_back("/usr/lib/");
    pfxs.push_back("/usr/local/lib/");
    char buffer[5];

    // hahaha! grossness is like a brother to us
#ifdef __APPLE__
    for (size_t pfxidx = 0; pfxidx < pfxs.size(); ++pfxidx) {
//        cerr << "prefix: " << pfxs[pfxidx] << endl;
	if (tryPreload(pfxs[pfxidx] + string("Python"))) return true;
        for (int minor = 8; minor >= 0; --minor) {
            sprintf(buffer, "%d", minor);
            if (tryPreload(pfxs[pfxidx] + string("libpython") + shortver + ".dylib." + buffer)) return true;
        }
        if (tryPreload(pfxs[pfxidx] + string("libpython") + shortver + ".dylib")) return true;
        if (tryPreload(pfxs[pfxidx] + string("libpython.dylib"))) return true;
    }
#else
    for (size_t pfxidx = 0; pfxidx < pfxs.size(); ++pfxidx) {
        for (int minor = 8; minor >= 0; --minor) {
            sprintf(buffer, "%d", minor);
            if (tryPreload(pfxs[pfxidx] + string("libpython") + shortver + ".so." + buffer)) return true;
        }
        if (tryPreload(pfxs[pfxidx] + string("libpython") + shortver + ".so")) return true;
        if (tryPreload(pfxs[pfxidx] + string("libpython.so"))) return true;
    }
#endif
        
    return false;
}


static PyExtensionManager pyExtensionManager;

const VampPluginDescriptor 
*vampGetPluginDescriptor(unsigned int version,unsigned int index)
{	
    if (version < 1) return 0;

	int isPythonInitialized = Py_IsInitialized();
	cerr << "# isPythonInitialized: " << isPythonInitialized << endl;
	cerr << "# haveScannedPlugins: " << haveScannedPlugins << endl;

	if (!haveScannedPlugins) {

		if (!isPythonInitialized){

			if (!preloadPython())
				cerr << "Warning: Could not preload Python. Dynamic loading in scripts will fail." << endl;
			if (PyImport_AppendInittab("vampy",initvampy) != 0)
				cerr << "Warning: Extension module could not be added to module inittab." << endl;
			Py_Initialize();
			array_API_initialiser();
			initvampy();
#ifdef _DEBUG			
		    cerr << "# isPythonInitialized after initialize: " << Py_IsInitialized() << endl;
#endif		
		}

		vector<string> pyPlugs;
		vector<string> pyPath;
		vector<PyObject *> pyClasses;
		static PyPlugScanner *scanner;
		
		//Scanning Plugins
		cerr << "Scanning Vampy Plugins" << endl;
		scanner = PyPlugScanner::getInstance();

		// added env. varable support VAMPY_EXTPATH
		pyPath=scanner->getAllValidPath();
		scanner->setPath(pyPath);
		
		// added env. variable support: 
		// VAMPY_COMPILED=1 to recognise .pyc files (default is 1)
		pyPlugs = scanner->getPyPlugs();

		cerr << "Found " << pyPlugs.size() << " Scripts." << endl;
		//TODO: should this support multiple classes per script (?)
		pyClasses = scanner->getPyClasses();
		cerr << "Found " << pyClasses.size() << " Classes." << endl;

		for (size_t i = 0; i < pyPlugs.size(); ++i) {
			adapters.push_back( new PyPluginAdapter(pyPlugs[i],pyClasses[i]));
		}
		pyExtensionManager.setPlugModuleNames(pyPlugs);
		pyExtensionManager.initExtension();
		array_API_initialiser();
		haveScannedPlugins=true;
	}

#ifdef _DEBUG
	cerr << "Accessing adapter index: " << index << " (adapters: " << adapters.size() << ")" << endl;
#endif	

	if (index<adapters.size()) {

		const VampPluginDescriptor *tmp = adapters[index]->getDescriptor();

		if (adapters[index]->failed()) { 
			cerr << "\nERROR: [in vampGetPluginDescriptor] Removing adapter of: \n'" 
			<< adapters[index]->getPlugKey() << "'\n" 
			<< "The plugin has failed to construct. Hint: Check __init__() function." << endl;
			pyExtensionManager.deleteModuleName(adapters[index]->getPlugKey());
			delete adapters[index];
			adapters.erase(adapters.begin()+index);
			return 0;
		}

		return tmp;

	} else return 0;
}








