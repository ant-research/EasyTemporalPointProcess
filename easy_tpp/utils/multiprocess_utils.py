import os
import time


def is_master_process():
    """ Check if the process is the master process in all machines.

    Returns:
        bool
    """
    rank = 0 if os.getenv('RANK') is None else int(os.getenv('RANK'))
    if rank == 0:
        return True
    else:
        return False


def is_local_master_process():
    """ Check if the process is the master process in the local machine.

    Returns:
        bool
    """
    rank = 0 if os.getenv('RANK') is None else int(os.getenv('RANK'))
    local_world_size = 1 if os.getenv('LOCAL_WORLD_SIZE') is None else int(os.getenv('LOCAL_WORLD_SIZE'))
    if local_world_size == 0 or rank % local_world_size == 0:
        return True
    else:
        return False


def get_now_timestamp_id():
    """ Get the current timestamp string.

    Returns:
        A string like yyMMdd_hhmmss
    """
    import datetime
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def get_unique_id():
    """ Generate a unique id string based on process id (pid), thread id and timestamp.

    Returns:
        Unique id: str
    """
    import os
    import threading
    pid = os.getpid()
    tid = threading.currentThread().ident
    ts_id = get_now_timestamp_id()

    return '{}_{}_{}'.format(pid, tid, ts_id)


def parse_uri_to_protocol_and_path(uri):
    """ Parse a uri into two parts, protocol and path. Set 'file' as default protocol when lack protocol.

    Args:
        uri: str
            The uri to identify a resource, whose format is like 'protocol://uri'.

    Returns:
        Protocol: str. The method to access the resource.
        URI: str. The location of the resource.
    """

    if uri is None:
        return None, None
    tokens = uri.split('://')
    if len(tokens) == 2:
        protocol = tokens[0]
        path = tokens[1]
    elif len(tokens) == 1:
        protocol = 'file'
        path = tokens[0]
    else:
        raise RuntimeError(f'Wrong url format: {uri}')

    return protocol, path


class Timer:
    """Count the elapsing time between start and end.
    """

    def __init__(self, unit='m'):
        unit = unit.lower()
        if unit == 's':
            self._unit = 1
        elif unit == 'm':
            self._unit = 60
        elif unit == 'h':
            self._unit = 1440
        else:
            raise RuntimeError('Unknown unit:', unit)

        self.unit = unit
        # default start time is set to the time the object initialized
        self._start_time = time.time()

    def start(self):
        self._start_time = time.time()

    def end(self):
        end_time = time.time()
        cost = (end_time - self._start_time) / self._unit
        # reset the start time using the end time
        self._start_time = end_time
        return '%.3f%s' % (cost, self.unit)


# -------------------------- Singleton Object --------------------------
default_timer = Timer()
