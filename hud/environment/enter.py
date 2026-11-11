"""Enter a workspace sandbox and run a command there, dropping privileges on the way in.

Started by ``entry_argv`` from the server's
interpreter: everything this process does before its final exec comes from the
server's filesystem, so a view built from a task image needs to supply nothing
but the command itself. A ``setpriv`` run inside the view would need util-linux
in the image, which busybox, alpine and distroless images lack.

It starts once per sandboxed process, so it imports only what it uses.
"""

import ctypes
import errno
import os
import signal
import sys

# The namespaces a process can join, in the order util-linux nsenter joins them.
_NAMESPACES = ("ipc", "uts", "net", "pid", "mnt")
# sys_admin, net_admin and net_raw: namespace root must not be able to inspect
# or redirect the authenticated verifier route while both briefly share its network.
_DROPPED_CAPABILITIES = (21, 12, 13)
_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38


def _check(libc: ctypes.CDLL, result: int) -> None:
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _join(libc: ctypes.CDLL, descriptors: dict[str, int]) -> None:
    """Join every namespace in ``descriptors``, the user namespace the way nsenter does.

    Joined first, a user namespace leaves no capability to join the others;
    joined last, it may be the one that grants them. So join the others
    where they already can be, then the user namespace, then the rest.
    """
    pending = [
        kind
        for kind in _NAMESPACES
        if kind in descriptors and libc.setns(descriptors[kind], 0) != 0
    ]
    if "user" in descriptors:
        _check(libc, libc.setns(descriptors["user"], 0))
    for kind in pending:
        _check(libc, libc.setns(descriptors[kind], 0))


def _continue_as_child() -> None:
    """Fork, so the command starts in the joined pid namespace; relay its end.

    Joining a pid namespace places only children in it.
    """
    child = os.fork()
    if child == 0:
        return
    while True:
        _, status = os.waitpid(child, os.WUNTRACED)
        if not os.WIFSTOPPED(status):
            break
        os.kill(os.getpid(), signal.SIGSTOP)
        os.kill(child, signal.SIGCONT)
    if os.WIFSIGNALED(status):
        signal.signal(os.WTERMSIG(status), signal.SIG_DFL)
        os.kill(os.getpid(), os.WTERMSIG(status))
    os._exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)


def main() -> None:
    assert sys.platform == "linux"
    separator = sys.argv.index("--")
    options, argv = sys.argv[1:separator], sys.argv[separator + 1 :]
    libc = ctypes.CDLL(None, use_errno=True)
    # Every path is opened before anything is joined: /proc/self/fd and
    # /proc/<pid> name things in the namespaces this process starts in.
    descriptors: dict[str, int] = {}
    drop_capabilities = no_new_privs = False
    identity: tuple[int, int] | None = None
    values = iter(options)
    for option in values:
        if option == "--enter":
            kind, _, path = next(values).partition("=")
            descriptors[kind] = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        elif option == "--drop-capabilities":
            drop_capabilities = True
        elif option == "--no-new-privs":
            no_new_privs = True
        elif option == "--identity":
            uid, _, gid = next(values).partition(":")
            identity = (int(uid), int(gid))
        else:
            raise SystemExit(f"hud-enter: unknown option {option}")
    cwd = descriptors.pop("cwd", None)
    _join(libc, descriptors)
    if cwd is not None:
        os.fchdir(cwd)
    if drop_capabilities:
        for capability in _DROPPED_CAPABILITIES:
            _check(libc, libc.prctl(_PR_CAPBSET_DROP, ctypes.c_ulong(capability), 0, 0, 0))
    if no_new_privs:
        _check(libc, libc.prctl(_PR_SET_NO_NEW_PRIVS, ctypes.c_ulong(1), 0, 0, 0))
    if identity is not None:
        os.setgroups([])
        os.setresgid(identity[1], identity[1], identity[1])
        os.setresuid(identity[0], identity[0], identity[0])
    for descriptor in (*descriptors.values(), *(() if cwd is None else (cwd,))):
        os.close(descriptor)
    if "pid" in descriptors:
        _continue_as_child()
    try:
        os.execvp(argv[0], argv)  # noqa: S606 - exact workspace-built argv
    except OSError as exc:
        sys.stderr.write(f"hud-enter: failed to execute {argv[0]}: {exc.strerror}\n")
        os._exit(126 if exc.errno in (errno.EACCES, errno.ENOEXEC) else 127)


if __name__ == "__main__":
    main()
