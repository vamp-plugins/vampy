/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/**
 * This Vamp plugin is a wrapper for Python Scripts. (VamPy)
 * Centre for Digital Music, Queen Mary, University of London.
 * Copyright 2008, George Fazekas.
 */

#include <Python.h>
#include "vamp/vamp.h"
#include "vamp-sdk/PluginAdapter.h"
#include "PyPlugScanner.h"
#include "PyPlugin.h"

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

//volatile bool mutex = false;
static int adinstcount;

class PyPluginAdapter : public Vamp::PluginAdapterBase
{
public: 
    PyPluginAdapter(std::string pyPlugId, PyObject* pyClass) :
        PluginAdapterBase(),
        m_plug(pyPlugId),		
        m_pyClass(pyClass)
    { 
        cerr << "PyPluginAdapter:ctor:"<< adinstcount << ": " << m_plug << endl; 
        adinstcount++;
        m_instanceCount = 0;
    }
    
    ~PyPluginAdapter() 
    {
    }
    
protected:
    Vamp::Plugin *createPlugin(float inputSampleRate)
    {
        try {
            PyPlugin *plugin = new PyPlugin(m_plug, inputSampleRate, m_pyClass);
            m_instanceCount++;
            return plugin;
        } catch (...) {
            cerr << "PyPluginAdapter::createPlugin: Failed to construct PyPlugin" << endl;
            return 0;
        }
    }
    
    std::string m_plug;
    bool m_haveInitialized;
    PyObject *m_pyClass;
    int m_instanceCount;
};


static std::vector<PyPluginAdapter *> adapters;
static bool haveScannedPlugins = false;

static bool tryPreload(string name)
{
    cerr << "Trying to load Python interpreter library \"" << name << "\"...";
#ifdef _WIN32
    void *lib = LoadLibrary(name.c_str());
    if (!lib) {
        cerr << " failed" << endl;
        return false;
    }
#else
    void *lib = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        cerr << " failed" << endl;
        return false;
    }
#endif
    cerr << " succeeded" << endl;
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

/* This doesn't work: don't try it again.
static bool initPython()
{
	// preloadPython();
	Py_Initialize();
#ifndef _WIN32
	//set dlopen flags form Python 
	string pyCmd = "from sys import setdlopenflags\nimport dl\nsetdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)\n";
	if (PyRun_SimpleString(pyCmd.c_str()) == -1) 
	{   
	    cerr << "Warning: Could not set dlopen flasgs. Dynamic loading in scripts will fail." << endl;
		return false;
	}
#endif	
	PyEval_InitThreads();			
	return Py_IsInitialized();
}    
*/

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
			Py_Initialize();
		    cerr << "# isPythonInitialized after initialize: " << Py_IsInitialized() << endl;
			// PyEval_InitThreads(); //not sure why this was needed
		}

		vector<string> pyPlugs;
		vector<string> pyPath;
		vector<PyObject *> pyClasses;
		static PyPlugScanner *scanner;
		
		//Scanning Plugins
		cerr << "Scanning PyPlugins" << endl;
		scanner = PyPlugScanner::getInstance();
		pyPath=scanner->getAllValidPath();
		//add this as extra path for development
		//pyPath.push_back("/Users/Shared/Development/vamp-experiments");
		scanner->setPath(pyPath);
		pyPlugs = scanner->getPyPlugs();
		cerr << "Found " << pyPlugs.size() << " Scripts ...OK" << endl;
		//TODO: this will support multiple classes per script
		pyClasses = scanner->getPyClasses();
		cerr << "Found " << pyClasses.size() << " Classes ...OK" << endl;

		for (size_t i = 0; i < pyPlugs.size(); ++i) {
			adapters.push_back( new PyPluginAdapter(pyPlugs[i],pyClasses[i]));
		} 
		haveScannedPlugins=true;		
		
	}

	cerr << "Accessing adapter index: " << index << " (adapters: " << adapters.size() << ")" << endl;
	if (index<adapters.size()) {
		const VampPluginDescriptor *tmp = adapters[index]->getDescriptor();
		return tmp;
	} else return 0;

	
}








