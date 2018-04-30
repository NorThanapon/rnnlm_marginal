import os
import json
import logging as py_logging

_log_level = {None: py_logging.NOTSET, 'debug': py_logging.DEBUG,
              'info': py_logging.INFO, 'warning': py_logging.WARNING,
              'error': py_logging.ERROR, 'critical': py_logging.CRITICAL}


def get_logger(log_file_path=None, name='default_log', level=None):
    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, py_logging.FileHandler):
                handler.baseFilename
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            '%(asctime)s [%(levelname)-5.5s] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S')
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = '\x1b[36m[%(levelname)-5.5s]\x1b[0m'
    log_formatter = py_logging.Formatter(f'{level_format}%(message)s')
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger


def time_span_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f'{int(h)}h {int(m)}m {s:2.4}s'


def ensure_dir(directory, delete=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif delete:
        backup_dir = f'{directory.rstrip("/")}_backup'
        current = backup_dir
        i = 1
        while os.path.exists(current):
            current = f'{backup_dir}_{i}'
            i += 1
        os.rename(directory, current)
        os.makedirs(directory)


def dump_opt(opt, logger, name=None, fpath=None):
    _d = dict(opt)
    if name is None and fpath is not None:
        name = os.path.basename(fpath)
    elif name is None:
        name = ''
    logger.info(f'{name}:\n{json.dumps(_d, indent=4, sort_keys=True)}')
    if fpath is not None:
        with open(fpath, 'w') as fp:
            json.dump(_d, fp, indent=4, sort_keys=True)


def nested_map(fn, maybe_structure, *args):
    if isinstance(maybe_structure, (list, tuple)):
        structure = maybe_structure
        output = []
        for maybe_structure in zip(structure, *args):
            output.append(nested_map(fn, *maybe_structure))
        try:
            return type(structure)(output)
        except TypeError:
            return type(structure)(*output)
    else:
        return fn(maybe_structure, *args)


def flatten(maybe_structure):
    _collect = []
    if isinstance(maybe_structure, (list, tuple)):
        structure = maybe_structure
        for maybe_structure in structure:
            _collect.extend(flatten(maybe_structure))
    else:
        _collect.append(maybe_structure)
    return _collect
