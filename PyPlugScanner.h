/*
    Vamp

    An API for audio analysis and feature extraction plugins.

    Centre for Digital Music, Queen Mary, University of London.
    Copyright 2006-2007 Chris Cannam and QMUL.
  
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


/*
	Objective: We want to find available pyVamp plugins here
	Future: We may have multiple plugins per script
*/

#ifndef _VAMP_PYPLUG_SCANNER_H_
#define _VAMP_PYPLUG_SCANNER_H_

#include "/usr/include/python/Python.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
//#include <fstream>
	
class PyPlugScanner
{
public:
	~PyPlugScanner() { m_hasInstance = false; }
	static PyPlugScanner *getInstance();	
	std::vector<std::string> getPyPlugs();
	std::vector<PyObject*> getPyInstances();
	void setPath(std::vector<std::string> path);
	std::vector<std::string> getAllValidPath();
	
protected:
	PyPlugScanner();
	PyObject *getScriptInstance(std::string path, std::string classname);
	std::vector<std::string> listFiles(std::string dir, std::string ext);
	static bool m_hasInstance;
	static PyPlugScanner *m_instance;
	std::string m_dir;
	std::vector<std::string> m_path; 
	std::vector<PyObject*> m_pyInstances;
};

#endif	
	