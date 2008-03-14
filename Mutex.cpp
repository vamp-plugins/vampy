/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
   Basic cross-platform mutex abstraction class.
   This file copyright 2007 Chris Cannam.
*/

#include "Mutex.h"

#include <iostream>

#ifndef _WIN32
#include <sys/time.h>
#include <time.h>
#endif

using std::cerr;
using std::endl;
using std::string;

#ifdef _WIN32

Mutex::Mutex()
#ifndef NO_THREAD_CHECKS
    :
    m_lockedBy(-1)
#endif
{
    m_mutex = CreateMutex(NULL, FALSE, NULL);
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)GetCurrentThreadId() << ": Initialised mutex " << &m_mutex << endl;
#endif
}

Mutex::~Mutex()
{
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)GetCurrentThreadId() << ": Destroying mutex " << &m_mutex << endl;
#endif
    CloseHandle(m_mutex);
}

void
Mutex::lock()
{
#ifndef NO_THREAD_CHECKS
    DWORD tid = GetCurrentThreadId();
    if (m_lockedBy == tid) {
        cerr << "ERROR: Deadlock on mutex " << &m_mutex << endl;
    }
#endif
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)tid << ": Want to lock mutex " << &m_mutex << endl;
#endif
    WaitForSingleObject(m_mutex, INFINITE);
#ifndef NO_THREAD_CHECKS
    m_lockedBy = tid;
#endif
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)tid << ": Locked mutex " << &m_mutex << endl;
#endif
}

void
Mutex::unlock()
{
#ifndef NO_THREAD_CHECKS
    DWORD tid = GetCurrentThreadId();
    if (m_lockedBy != tid) {
        cerr << "ERROR: Mutex " << &m_mutex << " not owned by unlocking thread" << endl;
        return;
    }
#endif
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)tid << ": Unlocking mutex " << &m_mutex << endl;
#endif
#ifndef NO_THREAD_CHECKS
    m_lockedBy = -1;
#endif
    ReleaseMutex(m_mutex);
}

bool
Mutex::trylock()
{
#ifndef NO_THREAD_CHECKS
    DWORD tid = GetCurrentThreadId();
#endif
    DWORD result = WaitForSingleObject(m_mutex, 0);
    if (result == WAIT_TIMEOUT || result == WAIT_FAILED) {
#ifdef DEBUG_MUTEX
        cerr << "MUTEX DEBUG: " << (void *)tid << ": Mutex " << &m_mutex << " unavailable" << endl;
#endif
        return false;
    } else {
#ifndef NO_THREAD_CHECKS
        m_lockedBy = tid;
#endif
#ifdef DEBUG_MUTEX
        cerr << "MUTEX DEBUG: " << (void *)tid << ": Locked mutex " << &m_mutex << " (from trylock)" << endl;
#endif
        return true;
    }
}

#else  /* !_WIN32 */

Mutex::Mutex()
#ifndef NO_THREAD_CHECKS
    :
    m_lockedBy(0),
    m_locked(false)
#endif
{
    pthread_mutex_init(&m_mutex, 0);
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)pthread_self() << ": Initialised mutex " << &m_mutex << endl;
#endif
}

Mutex::~Mutex()
{
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)pthread_self() << ": Destroying mutex " << &m_mutex << endl;
#endif
    pthread_mutex_destroy(&m_mutex);
}

void
Mutex::lock()
{
#ifndef NO_THREAD_CHECKS
    pthread_t tid = pthread_self();
    if (m_locked && m_lockedBy == tid) {
        cerr << "ERROR: Deadlock on mutex " << &m_mutex << endl;
    }
#endif
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)tid << ": Want to lock mutex " << &m_mutex << endl;
#endif
    pthread_mutex_lock(&m_mutex);
#ifndef NO_THREAD_CHECKS
    m_lockedBy = tid;
    m_locked = true;
#endif
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)tid << ": Locked mutex " << &m_mutex << endl;
#endif
}

void
Mutex::unlock()
{
#ifndef NO_THREAD_CHECKS
    pthread_t tid = pthread_self();
    if (!m_locked) {
        cerr << "ERROR: Mutex " << &m_mutex << " not locked in unlock" << endl;
        return;
    } else if (m_lockedBy != tid) {
        cerr << "ERROR: Mutex " << &m_mutex << " not owned by unlocking thread" << endl;
        return;
    }
#endif
#ifdef DEBUG_MUTEX
    cerr << "MUTEX DEBUG: " << (void *)tid << ": Unlocking mutex " << &m_mutex << endl;
#endif
#ifndef NO_THREAD_CHECKS
    m_locked = false;
#endif
    pthread_mutex_unlock(&m_mutex);
}

bool
Mutex::trylock()
{
#ifndef NO_THREAD_CHECKS
    pthread_t tid = pthread_self();
#endif
    if (pthread_mutex_trylock(&m_mutex)) {
#ifdef DEBUG_MUTEX
        cerr << "MUTEX DEBUG: " << (void *)tid << ": Mutex " << &m_mutex << " unavailable" << endl;
#endif
        return false;
    } else {
#ifndef NO_THREAD_CHECKS
        m_lockedBy = tid;
        m_locked = true;
#endif
#ifdef DEBUG_MUTEX
        cerr << "MUTEX DEBUG: " << (void *)tid << ": Locked mutex " << &m_mutex << " (from trylock)" << endl;
#endif
        return true;
    }
}

#endif
