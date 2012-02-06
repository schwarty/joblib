import os
import time
import warnings
import functools
import shutil

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


class SimpleStore(object):
    
    def __init__(self, func, cachedir, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None):
        
        self.func = func
        self._verbose = verbose

        if cachedir is None:
            self.cachedir = None
        else:
            self.cachedir = os.path.join(cachedir, 'joblib')
            mkdirp(self.cachedir)

        self.cachedir = cachedir
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
        mkdirp(self.cachedir)
        try:
            functools.update_wrapper(self, func)
        except:
            " Objects like ufunc don't like that "

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
            self._check_previous_func_code(stacklevel=3)
            and os.path.exists(output_dir)
            )

    @classmethod
    def reset_all(cls, cachedir):
        rm_subdirs(cachedir)

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
        func_code_file = os.path.join(func_dir, 'func_code.py')
        self._write_func_code(func_code_file, func_code, first_line)

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

    def _write_func_code(self, filename, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        func_code = '%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        with open(filename, 'w') as out:
            out.write(func_code)

    def _check_previous_func_code(self, stacklevel=2):
        """
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        # Here, we go through some effort to be robust to dynamically
        # changing code and collision. We cannot inspect.getsource
        # because it is not reliable when using IPython's magic "%run".
        func_code, source_file, first_line = get_func_code(self.func)
        func_dir = self._get_func_dir()
        func_code_file = os.path.join(func_dir, 'func_code.py')

        try:
            with open(func_code_file) as infile:
                old_func_code, old_first_line = \
                            extract_first_line(infile.read())
        except IOError:
                self._write_func_code(func_code_file, func_code, first_line)
                return False
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
