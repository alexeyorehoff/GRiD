"""Ownership and cancellation of native test/benchmark subprocess trees."""
import os
import signal
import subprocess


class _WindowsProcessTree:
    def __init__(self):
        self.jobs = []

    def Close(self):
        for job in self.jobs:
            job.Close()
        self.jobs.clear()


def start_process_tree(command, **kwargs):
    """Own the child before it can run or create an unowned descendant."""
    if os.name == "nt":
        import win32process
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | win32process.CREATE_SUSPENDED
    kwargs.setdefault("start_new_session", True)
    process = None
    job = None
    try:
        process = subprocess.Popen(command, **kwargs)
        job = own_process_tree(process)
        if os.name == "nt":
            import psutil
            import win32api
            import win32con
            import win32process
            # Popen closes CreateProcess's primary-thread handle. The suspended
            # process has not executed yet, so its only thread is the primary.
            thread_id = psutil.Process(process.pid).threads()[0].id
            thread = win32api.OpenThread(win32con.THREAD_SUSPEND_RESUME, False, thread_id)
            try:
                win32process.ResumeThread(thread)
            finally:
                thread.Close()
        return process, job
    except BaseException:
        try:
            if process is not None and process.poll() is None:
                if os.name == "nt" and job is None:
                    # Adoption failed while the child was still suspended.
                    process.kill()
                    process.wait()
                else:
                    terminate_process_tree(process, job)
        finally:
            if job is not None:
                job.Close()
        raise


def own_process_tree(process):
    """A Windows job kills descendants even if its owning launcher is killed."""
    if os.name != "nt":
        return None
    import win32api
    import win32con
    import win32job
    import psutil
    import uuid
    tree = _WindowsProcessTree()
    # A Windows venv launcher can have created the real Python child before
    # Popen returned. Adopt those descendants too; later children inherit the job.
    owned = [psutil.Process(process.pid)]
    owned.extend(owned[0].children(recursive=True))
    for member in owned:
        try:
            handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE |
                                         win32con.PROCESS_QUERY_INFORMATION, False, member.pid)
        except Exception:
            if not member.is_running():
                continue
            raise
        try:
            if any(win32job.IsProcessInJob(handle, job) for job in tree.jobs):
                continue
            # Already-running siblings can belong to different inherited job
            # hierarchies. Windows permits adding each to a fresh empty job.
            job = win32job.CreateJobObject(None, "grid-test-" + uuid.uuid4().hex)
            limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
            limits['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
            win32job.AssignProcessToJobObject(job, handle)
            tree.jobs.append(job)
        finally:
            handle.Close()
    return tree


def terminate_process_tree(process, job=None, timeout=5):
    if job is not None:
        import win32job
        for owned in job.jobs:
            win32job.TerminateJobObject(owned, 1)
        process.wait(timeout=timeout)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
