/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

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
	PyPluginAdapter(std::string pyPlugId, PyObject* pyInstance) :
		PluginAdapterBase(),
		m_plug(pyPlugId),		
		m_pyInstance(pyInstance)
		{ 
			cerr << "PyPluginAdapter:ctor:"<< adinstcount << ": " << m_plug << endl; 
			adinstcount++;
			m_instanceCount = 0;
		}
	
	~PyPluginAdapter() 
	{
	}

protected:
	Vamp::Plugin *createPlugin(float inputSampleRate) {
        
		std::string pclass = m_plug.substr(m_plug.rfind(':')+1,m_plug.size()-1);
		std::string ppath = m_plug.substr(0,m_plug.rfind(pathsep));
		PyPlugin *plugin = new PyPlugin(m_plug,inputSampleRate,m_pyInstance);
		m_instanceCount++;
		cerr << "PyPluginAdapter::createPlugin:" << pclass << " (instance: " << m_instanceCount << ")" << endl;
		return plugin;

		}

	std::string m_plug;
	bool m_haveInitialized;
	PyObject *m_pyInstance;
	int m_instanceCount;

};


static std::vector<PyPluginAdapter *> adapters;
static bool haveScannedPlugins = false;

static bool tryPreload(string name)
{
//    cerr << "Trying to load Python interpreter library \"" << name << "\"...";
    void *lib = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
//        char *err = dlerror();
//        if (err && err[0]) {
//            cerr << " failed (" << err << ")" << endl;
//        } else {
//            cerr << " failed" << endl;
//        }
        return false;
    }
//    cerr << " succeeded" << endl;
    return true;
}

static bool preloadPython()
{
    // useless on Windows

    string pyver = Py_GetVersion();
    int dots = 2;
    string shortver;
    for (int i = 0; i < pyver.length(); ++i) {
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
    for (int pfxidx = 0; pfxidx < pfxs.size(); ++pfxidx) {
        for (int minor = 8; minor >= 0; --minor) {
            sprintf(buffer, "%d", minor);
            if (tryPreload(pfxs[pfxidx] + string("libpython") + shortver + ".dylib." + buffer)) return true;
        }
        if (tryPreload(pfxs[pfxidx] + string("libpython") + shortver + ".dylib")) return true;
        if (tryPreload(pfxs[pfxidx] + string("libpython.dylib"))) return true;
    }
#else
    for (int pfxidx = 0; pfxidx < pfxs.size(); ++pfxidx) {
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
    

const VampPluginDescriptor 
*vampGetPluginDescriptor(unsigned int version,unsigned int index)
{	
    if (version < 1) return 0;

	int isPythonInitialized = Py_IsInitialized();
	//cerr << "# isPythonInitialized: " << isPythonInitialized << endl;
	//cerr << "# haveScannedPlugins: " << haveScannedPlugins << endl;

	if (!haveScannedPlugins) {

		if (!isPythonInitialized) {

			if (!preloadPython()) {
                            cerr << "Warning: Could not preload Python." 
                                 << " Dynamic loading in scripts will fail." << endl;
                        }

/*
			void *pylib = 0; 
			
			cerr << "Loading Python Interpreter at: " << pythonPath << endl;
			//Preloading the binary allows the load of shared libs 
			//TODO: check how to do RTLD_NOW on Windows
#ifdef _WIN32
			pylib = LoadLibrary(pythonPath.c_str());
#else			
			pylib = dlopen(pythonPath.c_str(), RTLD_NOW|RTLD_GLOBAL);
#endif			
			if (!pylib) cerr << "Warning: Could not preload Python." 
						<< " Dynamic loading in scripts will fail." << endl;
*/
			Py_Initialize();
	 		PyEval_InitThreads();			
		} else {
			//Py_InitializeEx(1);
		}

		vector<string> pyPlugs;
		vector<string> pyPath;
		vector<PyObject *> pyInstances;
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
		pyInstances = scanner->getPyInstances();
		cerr << "Found " << pyInstances.size() << " Instances ...OK" << endl;

		for (size_t i = 0; i < pyPlugs.size(); ++i) {
			adapters.push_back( new PyPluginAdapter(pyPlugs[i],pyInstances[i]));
		} 
		haveScannedPlugins=true;		
		
	}

	cerr << "Accessing adapter index: " << index << " (adapters: " << adapters.size() << ")" << endl;
	if (index<adapters.size()) {
		const VampPluginDescriptor *tmp = adapters[index]->getDescriptor();
		return tmp;
	} else return 0;

	
}








