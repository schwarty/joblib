"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.


from __future__ import with_statement
import os
import shutil
import sys
import time
import pydoc
try:
    import cPickle as pickle
except ImportError:
    import pickle
import functools
import traceback
import warnings
import inspect
try:
    # json is in the standard library for Python >= 2.6
    import json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        # Not the end of the world: we'll do without this functionality
        json = None

# Local imports
from .hashing import hash
from .func_inspect import get_func_code, get_func_name, filter_args
from .logger import Logger, format_time
from . import numpy_pickle
from .disk import mkdirp, rm_subdirs
from .store import SimpleStore


# TODO: The following object should have a data store object as a sub
# object, and the interface to persist and query should be separated in
# the data store.
#
# This would enable creating 'Memory' objects with a different logic for
# pickling that would simply span a MemorizedFunc with the same
# store (or do we want to copy it to avoid cross-talks?), for instance to
# implement HDF5 pickling.

# TODO: Same remark for the logger, and probably use the Python logging
# mechanism.




###############################################################################
# class `MemorizedFunc`
###############################################################################
class MemorizedFunc(Logger):
    """ Callable object decorating a function for caching its return value
        each time it is called.

        All values are cached on the filesystem, in a deep directory
        structure. Methods are provided to inspect the cache or clean it.

        Attributes
        ----------
        func: callable
            The original, undecorated, function.
        cachedir: string
            Path to the base cache directory of the memory context.
        ignore: list or None
            List of variable names to ignore when choosing whether to
            recompute.
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments.
        compress: boolean
            Whether to zip the stored data on disk. Note that compressed
            arrays cannot be read by memmapping.
        verbose: int, optional
            The verbosity flag, controls messages that are issued as
            the function is revaluated.
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------

    def __init__(self, func, cache_scheme, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None):
        """
            Parameters
            ----------
            func: callable
                The function to decorate
            cachedir: string
                The path of the base directory to use as a data store
            ignore: list or None
                List of variable names to ignore.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments.
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued
                as functions are revaluated. The higher, the more verbose
            timestamp: float, optional
                The reference time from which times in tracing messages
                are reported.
        """
        Logger.__init__(self)
        self._verbose = verbose
        self.cache_scheme = cache_scheme
        self.func = func
        self.mmap_mode = mmap_mode
        self.compress = compress
        if compress and mmap_mode is not None:
            warnings.warn('Compressed results cannot be memmapped',
                          stacklevel=2)
        if timestamp is None:
            timestamp = time.time()
        self.timestamp = timestamp
        if ignore is None:
            ignore = []
        self.ignore = ignore

        try:
            functools.update_wrapper(self, func)
        except:
            " Objects like ufunc don't like that "
        if inspect.isfunction(func):
            doc = pydoc.TextDoc().document(func
                                    ).replace('\n', '\n\n', 1)
        else:
            # Pydoc does a poor job on other objects
            doc = func.__doc__
        self.__doc__ = 'Memoized version of %s' % doc

        if isinstance(self.cache_scheme, (tuple, list)):
            self.cache = self.cache_scheme[0](func, 
                                              *self.cache_scheme[1:],
                                              mmap_mode=mmap_mode,
                                              ignore=ignore,
                                              compress=self.compress,
                                              verbose=verbose,
                                              timestamp=self.timestamp)
        else:
            self.cache = SimpleStore(func,
                                     self.cache_scheme,
                                     mmap_mode=mmap_mode,
                                     ignore=ignore,
                                     compress=self.compress,
                                     verbose=verbose,
                                     timestamp=self.timestamp)

    def __call__(self, *args, **kwargs):
        # Compare the function code with the previous to see if the
        # function code has changed
        # output_dir, _ = self.cache.get_output_dir(*args, **kwargs)
        # FIXME: The statements below should be try/excepted
        if not self.cache.exists(*args, **kwargs):
            return self.call(*args, **kwargs)
        else:
            try:
                t0 = time.time()
                out = self.cache.get(*args, **kwargs)
                if self._verbose > 4:
                    t = time.time() - t0
                    _, name = get_func_name(self.func)
                    msg = '%s cache loaded - %s' % (name, format_time(t))
                    print max(0, (80 - len(msg))) * '_' + msg
                return out
            except Exception:
                # XXX: Should use an exception logger
                self.warn('Exception while loading results for '
                          '(args=%s, kwargs=%s)\n %s' %
                          (args, kwargs, traceback.format_exc()))
                
                self.cache.delete(*args, **kwargs)
                return self.call(*args, **kwargs)

    def __reduce__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
            In addition, when unpickling, we run the __init__
        """
        return (self.__class__, (self.func, self.cache_scheme, self.ignore,
                self.mmap_mode, self.compress, self._verbose))

    def call(self, *args, **kwargs):
        """ Force the execution of the function with the given arguments and
            persist the output values.
        """
        start_time = time.time()
        if self._verbose:
            print self.format_call(*args, **kwargs)

        output = self.func(*args, **kwargs)
        self.cache.set(output, args, kwargs)

        # output_dir, argument_hash = self.cache.get_output_dir(*args, **kwargs)
        # output = self.func(*args, **kwargs)
        # self.cache._persist_output(output, output_dir)
        # input_repr = self.cache._persist_input(output_dir, *args, **kwargs)
        duration = time.time() - start_time
        if self._verbose:
            _, name = get_func_name(self.func)
            msg = '%s - %s' % (name, format_time(duration))
            print max(0, (80 - len(msg))) * '_' + msg
        return output

    def format_call(self, *args, **kwds):
        """ Returns a nicely formatted statement displaying the function
            call with the given arguments.
        """
        path, signature = self.format_signature(self.func, *args,
                            **kwds)
        msg = '%s\n[Memory] Calling %s...\n%s' % (80 * '_', path, signature)
        return msg
        # XXX: Not using logging framework
        #self.debug(msg)

    def format_signature(self, func, *args, **kwds):
        # XXX: This should be moved out to a function
        # XXX: Should this use inspect.formatargvalues/formatargspec?
        module, name = get_func_name(func)
        module = [m for m in module if m]
        if module:
            module.append(name)
            module_path = '.'.join(module)
        else:
            module_path = name
        arg_str = list()
        previous_length = 0
        for arg in args:
            arg = self.format(arg, indent=2)
            if len(arg) > 1500:
                arg = '%s...' % arg[:700]
            if previous_length > 80:
                arg = '\n%s' % arg
            previous_length = len(arg)
            arg_str.append(arg)
        arg_str.extend(['%s=%s' % (v, self.format(i)) for v, i in
                                    kwds.iteritems()])
        arg_str = ', '.join(arg_str)

        signature = '%s(%s)' % (name, arg_str)
        return module_path, signature

    # Make make public


    # XXX: Need a method to check if results are available.

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------

    def __repr__(self):
        return '%s(func=%s, cachedir=%s)' % (
                    self.__class__.__name__,
                    self.func,
                    repr(self.cache_scheme),
                    )

###############################################################################
# class `Memory`
###############################################################################
class Memory(Logger):
    """ A context object for caching a function's return value each time it
        is called with the same input arguments.

        All values are cached on the filesystem, in a deep directory
        structure.

        see :ref:`memory_reference`
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------

    def __init__(self, cache_scheme, mmap_mode=None, compress=False, verbose=1):
        """
            Parameters
            ----------
            cachedir: string or None
                The path of the base directory to use as a data store
                or None. If None is given, no caching is done and
                the Memory object is completely transparent.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments.
            compress: boolean
                Whether to zip the stored data on disk. Note that
                compressed arrays cannot be read by memmapping.
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued
                as functions are revaluated.
        """
        # XXX: Bad explaination of the None value of cachedir
        Logger.__init__(self)
        self._verbose = verbose
        self.mmap_mode = mmap_mode
        self.timestamp = time.time()
        self.compress = compress
        if compress and mmap_mode is not None:
            warnings.warn('Compressed results cannot be memmapped',
                          stacklevel=2)

        self.cache_scheme = cache_scheme

    def cache(self, func=None, ignore=None, verbose=None,
                        mmap_mode=False):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(self.cache, ignore=ignore)
        if self.cache_scheme is None:
            return func
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, MemorizedFunc):
            func = func.func
        return MemorizedFunc(func, cache_scheme=self.cache_scheme,
                                   mmap_mode=mmap_mode,
                                   ignore=ignore,
                                   compress=self.compress,
                                   verbose=verbose,
                                   timestamp=self.timestamp)

    def clear(self, warn=True):
        """ Erase the complete cache directory.
        """
        if warn:
            self.warn('Flushing completely the cache')

        # if isinstance(self.cache_scheme, (tuple, list)):
        #     cache = self.cache_scheme[0](None, *self.cache_scheme[1:])
        # else:
        #     cache = SimpleStore(None, self.cache_scheme)

        SimpleStore.reset_all(self.cache_scheme)

    def eval(self, func, *args, **kwargs):
        """ Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        """
        if self.cachedir is None:
            return func(*args, **kwargs)
        return self.cache(func)(*args, **kwargs)

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------

    def __repr__(self):
        return '%s(cachedir=%s)' % (
                    self.__class__.__name__,
                    repr(self.cache_scheme),
                    )

    def __reduce__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
            In addition, when unpickling, we run the __init__
        """
        # We need to remove 'joblib' from the end of cachedir
        return (self.__class__, (self.cache_scheme,
                self.mmap_mode, self.compress, self._verbose))
