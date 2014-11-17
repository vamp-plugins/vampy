
#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <iostream>
#include <cstdlib>

class MyDebug
{
public:
    MyDebug() : want(std::getenv("VAMPY_VERBOSE") != 0) { }

    template <typename T>
    MyDebug &operator<<(const T &t) {
	if (want) std::cerr << t;
        return *this;
    }
    
    MyDebug &operator<<(std::ostream &(*o)(std::ostream &)) {
	if (want) std::cerr << o;
        return *this;
    }
    
private:
    bool want;
};

#define DSTREAM (MyDebug())

#endif
