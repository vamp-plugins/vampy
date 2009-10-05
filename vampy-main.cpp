/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>

#ifdef HAVE_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API
#include "numpy/arrayobject.h"
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
            PyPlugin *plugin = new PyPlugin(m_plug, inputSampleRate, m_pyClass, totinstcount);
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
/// numpy C-API requirement
#ifdef HAVE_NUMPY
	import_array();
	if(NPY_VERSION != PyArray_GetNDArrayCVersion())
		cerr << "Warning: Numpy ABI version mismatch. (Build version: " 
		<< NPY_VERSION << " Runtime version: " << PyArray_GetNDArrayCVersion() << ")" << endl;
#endif
}


static std::vector<PyPluginAdapter *> adapters;
static bool haveScannedPlugins = false;

static bool tryPreload(string name)
{
#ifdef _WIN32
    void *lib = LoadLibrary(name.c_str());
    if (!lib) {
        return false;
    }
#else
    void *lib = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        return false;
    }
#endif
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

    vector<string> pfxs;
    pfxs.push_back("");
    pfxs.push_back(string(Py_GetExecPrefix()) + "/lib/");
    pfxs.push_back(string(Py_GetExecPrefix()) + "/");
    pfxs.push_back("/usr/lib/");
    pfxs.push_back("/usr/local/lib/");
    char buffer[5];

    // hahaha! grossness is like a brother to us
#ifdef __APPLE__
    for (size_t pfxidx = 0; pfxidx < pfxs.size(); ++pfxidx) {
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








