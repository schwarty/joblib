import os
import time
import warnings
import functools
import shutil
import tempfile

try:
    # json is in the standard library for Python >= 2.6
    import json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        # Not the end of the world: we'll do without this functionality
        json = None

from .func_inspect import get_func_code, get_func_name, filter_args
from .disk import mkdirp, rm_subdirs
from . import numpy_pickle
from .logger import Logger, format_time
from .hashing import hash

FIRST_LINE_TEXT = "# first line:"

def extract_first_line(func_code):
    """ Extract the first line information from the function code
        text if available.
    """
    if func_code.startswith(FIRST_LINE_TEXT):
        func_code = func_code.split('\n')
        first_line = int(func_code[0][len(FIRST_LINE_TEXT):])
        func_code = '\n'.join(func_code[1:])
    else:
        first_line = -1
    return func_code, first_line


class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """


class Store(object):

    def __init__(self, func=None, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None):
        
        self.func = func
        self._verbose = verbose

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

    def get(self, *args, **kwargs):
        raise NotImplemented

    def set(self, value, args=(), kwargs={}):
        raise NotImplemented

    def delete(self, *args, **kwargs):
        raise NotImplemented

    def exists(self, *args, **kwargs):
        raise NotImplemented

    def clear(self):
        raise NotImplemented

    def reset_all(self):
        raise NotImplemented

    def _get_func_code(self):
        raise NotImplemented

    def _save_func_code(self, func_code, first_line):
        raise NotImplemented

    def _is_same_func(self, stacklevel=2):
        """
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        # Here, we go through some effort to be robust to dynamically
        # changing code and collision. We cannot inspect.getsource
        # because it is not reliable when using IPython's magic "%run".

        func_code, source_file, first_line = get_func_code(self.func)
        old = self._get_func_code()

        if not old:
            self._save_func_code(func_code, first_line)
            return False
        else: 
            old_func_code, old_first_line = old

        if old_func_code == func_code:
            return True

        # We have differing code, is this because we are refering to
        # differing functions, or because the function we are refering as
        # changed?

        if old_first_line == first_line == -1:
            _, func_name = get_func_name(self.func, resolv_alias=False,
                                         win_characters=False)
            if not first_line == -1:
                func_description = '%s (%s:%i)' % (func_name,
                                                source_file, first_line)
            else:
                func_description = func_name
            warnings.warn(JobLibCollisionWarning(
                "Cannot detect name collisions for function '%s'"
                        % func_description), stacklevel=stacklevel)

        # Fetch the code at the old location and compare it. If it is the
        # same than the code store, we have a collision: the code in the
        # file has not changed, but the name we have is pointing to a new
        # code block.
        if (not old_first_line == first_line
                                    and source_file is not None
                                    and os.path.exists(source_file)):
            _, func_name = get_func_name(self.func, resolv_alias=False)
            num_lines = len(func_code.split('\n'))
            on_disk_func_code = file(source_file).readlines()[
                    old_first_line - 1:old_first_line - 1 + num_lines - 1]
            on_disk_func_code = ''.join(on_disk_func_code)
            if on_disk_func_code.rstrip() == old_func_code.rstrip():
                warnings.warn(JobLibCollisionWarning(
                'Possible name collisions between functions '
                "'%s' (%s:%i) and '%s' (%s:%i)" %
                (func_name, source_file, old_first_line,
                 func_name, source_file, first_line)),
                 stacklevel=stacklevel)

        # The function has changed, wipe the cache directory.
        # XXX: Should be using warnings, and giving stacklevel
        self.clear(warn=True)
        return False


class DiskStore(Store):
    
    def __init__(self, cachedir, func=None, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None):
        
        Store.__init__(self, func, ignore, mmap_mode,
                       compress, verbose, timestamp)

        if cachedir is None:
            self.cachedir = None
        else:
            _, cachedir = cachedir.split(':')
            self.cachedir = os.path.join(cachedir, 'joblib')
            mkdirp(self.cachedir)

    def get(self, *args, **kwargs):
        output_dir, _ = self.get_output_dir(*args, **kwargs)
        return self.load_output(output_dir)

    def set(self, value, args=(), kwargs={}):
        output_dir, argument_hash = self.get_output_dir(*args, **kwargs)
        self._persist_output(value, output_dir)
        self._persist_input(output_dir, *args, **kwargs)

    def delete(self, *args, **kwargs):
        output_dir, _ = self.get_output_dir(*args, **kwargs)
        shutil.rmtree(output_dir, ignore_errors=True)

    def exists(self, *args, **kwargs):
        output_dir, _ = self.get_output_dir(*args, **kwargs)
        return (
            self._is_same_func(stacklevel=3)
            and os.path.exists(output_dir)
            )

    def reset_all(self):
        rm_subdirs(self.cachedir)

    def clear(self, warn=True):
        """ Empty the function's cache.
        """
        func_dir = self._get_func_dir(mkdir=False)
        # if self._verbose and warn:
        #     self.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir, ignore_errors=True)
        mkdirp(func_dir)
        func_code, _, first_line = get_func_code(self.func)
        self._save_func_code(func_code, first_line)

    def _get_func_code(self):
        func_dir = self._get_func_dir()
        func_code_file = os.path.join(func_dir, 'func_code.py')

        try:
            with open(func_code_file) as infile:
                old_func_code, old_first_line = \
                    extract_first_line(infile.read())
        except IOError:
            return False

        return old_func_code, old_first_line

    def _save_func_code(self, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        func_dir = self._get_func_dir()
        func_code_file = os.path.join(func_dir, 'func_code.py')
        func_code = '%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        with open(func_code_file, 'w') as out:
            out.write(func_code)

    def _get_func_dir(self, mkdir=True):
        """ Get the directory corresponding to the cache for the
            function.
        """
        module, name = get_func_name(self.func)
        module.append(name)
        func_dir = os.path.join(self.cachedir, *module)
        if mkdir:
            mkdirp(func_dir)
        return func_dir

    def get_output_dir(self, *args, **kwargs):
        """ Returns the directory in which are persisted the results
            of the function corresponding to the given arguments.

            The results can be loaded using the .load_output method.
        """
        coerce_mmap = (self.mmap_mode is not None)
        argument_hash = hash(filter_args(self.func, self.ignore,
                             *args, **kwargs),
                             coerce_mmap=coerce_mmap)
        output_dir = os.path.join(self._get_func_dir(self.func),
                                  argument_hash)
        return output_dir, argument_hash

    def _persist_output(self, output, dir):
        """ Persist the given output tuple in the directory.
        """
        try:
            mkdirp(dir)
            filename = os.path.join(dir, 'output.pkl')
            numpy_pickle.dump(output, filename, compress=self.compress)
        except OSError:
            " Race condition in the creation of the directory "

    def _persist_input(self, output_dir, *args, **kwargs):
        """ Save a small summary of the call using json format in the
            output directory.
        """
        argument_dict = filter_args(self.func, self.ignore,
                                    *args, **kwargs)

        input_repr = dict((k, repr(v)) for k, v in argument_dict.iteritems())
        if json is not None:
            # This can fail do to race-conditions with multiple
            # concurrent joblibs removing the file or the directory
            try:
                mkdirp(output_dir)
                json.dump(
                    input_repr,
                    file(os.path.join(output_dir, 'input_args.json'), 'w'),
                    )
            except:
                pass
        return input_repr

    def load_output(self, output_dir):
        """ Read the results of a previous calculation from the directory
            it was cached in.
        """
        if self._verbose > 1:
            t = time.time() - self.timestamp
            print '[Memory]% 16s: Loading %s...' % (
                                    format_time(t),
                                    self.format_signature(self.func)[0]
                                    )
        filename = os.path.join(output_dir, 'output.pkl')
        return numpy_pickle.load(filename,
                                 mmap_mode=self.mmap_mode)


class MongoDBStore(Store):
    
    def __init__(self, uri, func=None, ignore=None, mmap_mode=None,
                     compress=False, verbose=1, timestamp=None):

        """
        Parameters
        ----------
        uri: str
        Simple URI scheme to connect to a mongodb database.
        e.g. mongodb://host:port/database
        """

        Store.__init__(self, func, ignore, mmap_mode,
                       compress, verbose, timestamp)

        # delayed import
        from pymongo import Connection
        import gridfs

        server, database = uri.split('/')[2:]
        host, port = server.split(':')
        module, name = get_func_name(self.func)
        func_col = '%s.%s' % (module[0], name)

        self.dbname = database
        self.db = Connection(host, int(port))[database]
        self.fs = gridfs.GridFS(self.db, collection=func_col)
        self.files = self.db['%s.files' % func_col]
        self.chunks = self.db['%s.chunks' % func_col]

    def get(self, *args, **kwargs):
        if self._verbose > 1:
            t = time.time() - self.timestamp
            print '[Memory]% 16s: Loading %s...' % (
                                    format_time(t),
                                    self.format_signature(self.func)[0]
                                    )
        module, name = get_func_name(self.func)
        module = module[0]
        arg_hash = self._argument_hash(*args, **kwargs)
        filename = '%s/%s/%s/output.pkl' % (module, name, arg_hash)

        tmppath = tempfile.mkstemp()[1]
        tmpfile = open(tmppath, 'w')
        tmpfile.write(
            self.fs.get(self.fs.get_last_version(filename)._id).read())
        tmpfile.close()

        ret = numpy_pickle.load(tmppath,
                                 mmap_mode=self.mmap_mode)

        os.remove(tmppath)
        return ret

    def set(self, value, args=(), kwargs={}):
        module, name = get_func_name(self.func)
        module = module[0]
        arg_hash = self._argument_hash(*args, **kwargs)
        filename = '%s/%s/%s/output.pkl' % (module, name, arg_hash)

        tmpfile = tempfile.mkstemp()[1]
        numpy_pickle.dump(value, tmpfile, compress=self.compress)

        self.fs.put(open(tmpfile), filename=filename)
        os.remove(tmpfile)

        filename = '%s/%s/%s/input_args.json' % (module, name, arg_hash)
        argument_dict = filter_args(self.func, self.ignore,
                                    *args, **kwargs)
        input_repr = dict((k, repr(v)) for k, v in argument_dict.iteritems())

        if json is not None:
            # This can fail do to race-conditions with multiple
            # concurrent joblibs removing the file or the directory
            try:
                self.fs.put(json.dumps(input_repr), filename=filename)
            except:
                pass

    def delete(self, *args, **kwargs):
        module, name = get_func_name(self.func)
        module = module[0]
        arg_hash = self._argument_hash(*args, **kwargs)
        filename = '%s/%s/%s/output.pkl' % (module, name, arg_hash)
        self.fs.delete(self.fs.get_last_version(filename)._id)
        filename = '%s/%s/%s/input_args.json' % (module, name, arg_hash)
        self.fs.delete(self.fs.get_last_version(filename)._id)

    def exists(self, *args, **kwargs):
        module, name = get_func_name(self.func)
        module = module[0]
        arg_hash = self._argument_hash(*args, **kwargs)
        filename = '%s/%s/%s/input_args.json' % (module, name, arg_hash)

        return (
            self._is_same_func(stacklevel=3)
            and self.fs.exists({'filename':filename})
            )

    def reset_all(self):
        self.mongo.drop(self.dbname)

    def clear(self, warn=True):
        """ Empty the function's cache.
        """

        self.files.drop()
        self.chunks.drop()

        func_code, _, first_line = get_func_code(self.func)
        self._save_func_code(func_code, first_line)

    def _argument_hash(self, *args, **kwargs):
        coerce_mmap = (self.mmap_mode is not None)
        argument_hash = hash(filter_args(self.func, self.ignore,
                                         *args, **kwargs),
                             coerce_mmap=coerce_mmap)

        return argument_hash

    def _get_func_code(self):
        module, name = get_func_name(self.func)
        module = module[0]
        filename = '%s/%s/%s' % (module, name, 'func_code.py')

        try:
            func_file = self.fs.get_last_version(filename)
            old_func_code, old_first_line =  \
                extract_first_line(func_file.read())
        except:
            return False

        return old_func_code, old_first_line

    def _save_func_code(self, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        module, name = get_func_name(self.func)
        module = module[0]
        func_code = '%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        filename = '%s/%s/%s' % (module, name, 'func_code.py')

        self.fs.put(func_code, filename=filename)


schemes = {
    'mongodb':MongoDBStore,
    '':DiskStore
    }

def store_from_scheme(store_scheme):
    if isinstance(store_scheme, (tuple, list)):
        return functools.partial(store_scheme[0], *store_scheme[1:])
    elif isinstance(store_scheme, (str, unicode)):
        scheme = store_scheme.split(':')[0]
        return functools.partial(schemes[scheme], store_scheme)
