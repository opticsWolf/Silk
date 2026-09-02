"""Sandbox configuration and path-safety helpers for file tools."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


class FileToolSandbox:
    """
    Per-session sandbox that constrains file-tool access.

    Read visibility is governed by ``allowed_paths`` (allowlist) and
    ``denied_paths`` (denylist): a path that is not visible is filtered out of
    listings/searches and cannot be read â€” it is effectively invisible to the
    LLM. Write access is governed by the global ``write_enabled`` flag and,
    optionally, a ``writable_paths`` allowlist for per-path read-only/read-write
    distinctions.

    Set ``enabled=False`` to switch path confinement off entirely (the tools
    then operate on the whole filesystem, subject only to OS permissions and the
    read/write/delete booleans). This is a deliberate, clearly-labelled escape
    hatch driven by the "Enable sandbox" toggle in the UI.
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        allowed_paths: Optional[list[str | Path]] = None,
        denied_paths: Optional[list[str | Path]] = None,
        max_read_bytes: int = 512 * 1024,       # 512 KiB
        max_write_bytes: int = 256 * 1024,       # 256 KiB
        read_enabled: bool = True,
        write_enabled: bool = True,
        delete_enabled: bool = False,
        allowed_extensions: Optional[list[str]] = None,
        dry_run: bool = False,
        enabled: bool = True,
        writable_paths: Optional[list[str | Path]] = None,
        path_modes: Optional[dict[str | Path, str]] = None,
    ):
        # Announce the root to the process-wide lock registry: a subprocess
        # that may write takes its root exclusively, and it can only do that
        # for roots it knows about (spec D67 tier 2). Imported here, like
        # lock_paths below, so the sandbox stays cheap to import.
        from .file_locks import register_root

        self.root_dir = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
        register_root(self.root_dir)
        self.allowed_paths = [
            Path(p).resolve() for p in (allowed_paths or [self.root_dir])
        ]
        self.denied_paths = [
            Path(p).resolve() for p in (denied_paths or [])
        ]
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.delete_enabled = delete_enabled
        self.allowed_extensions = allowed_extensions  # None = any extension
        self.dry_run = dry_run
        self.enabled = enabled
        # None = no per-path restriction (writability falls back to write_enabled).
        self.writable_paths = (
            [Path(p).resolve() for p in writable_paths]
            if writable_paths is not None else None
        )
        # Hierarchical access policy: a resolved-path → mode map where mode is
        # "read", "read_write", or "blocked". A directory rule covers its whole
        # subtree; a longer (more-specific) rule — e.g. a per-file override —
        # wins over an ancestor directory rule. When set, this drives read/
        # write resolution instead of allowed_paths/writable_paths, so files
        # created after selection (by the agent or dropped in by the user)
        # automatically inherit the mode of the nearest granted directory.
        self.path_modes: Optional[dict[Path, str]] = (
            {Path(p).resolve(): m for p, m in path_modes.items()}
            if path_modes is not None else None
        )

    # -- narrowing (spec D16/D18, invariant I6) ---------------------------

    def restrict(self, path_modes: dict, *, write: bool = True):
        """Tighten this sandbox for the duration of a ``with`` block.

        The Agent node applies the grant that reached it down the ToolSet →
        Role → Agent chain (D16). It has to happen **in place**: the file
        tools closed over this object when they were registered, so a
        replacement sandbox would be built and then ignored, and rebuilding
        the whole ToolBox would drop anything attached to it live (the
        orchestrator's delegation tools, the run's approval gate).

        Narrowing only, in the literal sense: every resulting mode is the
        lesser of what this sandbox already allowed and what was asked for,
        and a path this sandbox did not cover is not added (I6). Confinement
        itself is untouchable here -- ``enabled`` is a ToolBox-level choice
        and no grant can turn it back on (D18).

        Restores the previous policy on exit, because the sandbox is a live
        graph object shared with the next run.
        """
        from contextlib import contextmanager

        from ..file_grants import MODE_BLOCKED, lesser_mode

        @contextmanager
        def _scope():
            previous_modes = self.path_modes
            previous_write = self.write_enabled
            narrowed: dict[Path, str] = {}
            for raw, asked in (path_modes or {}).items():
                path = Path(raw).resolve()
                current = self.resolve_mode(path)
                if current is None and previous_modes is None:
                    # No per-path policy at all: the sandbox's own read/write
                    # flags are the ceiling.
                    current = "read_write" if self.write_enabled else "read"
                elif current is None:
                    # There *is* a policy and it does not cover this path, so
                    # neither does the narrowing. Widening here is exactly the
                    # failure I6 exists to prevent.
                    current = MODE_BLOCKED
                mode = lesser_mode(current, asked)
                if mode != MODE_BLOCKED:
                    narrowed[path] = mode
            # Explicit blocks carve holes and must survive the filter above.
            for raw, asked in (path_modes or {}).items():
                if asked == MODE_BLOCKED:
                    narrowed[Path(raw).resolve()] = MODE_BLOCKED
            self.path_modes = narrowed
            self.write_enabled = bool(
                write and previous_write
                and any(m == "read_write" for m in narrowed.values())
            )
            try:
                yield self
            finally:
                self.path_modes = previous_modes
                self.write_enabled = previous_write

        return _scope()

    # -- path resolution -------------------------------------------------

    def resolve_path(self, path_str: str) -> Path:
        """
        Resolve *path_str* relative to ``root_dir`` and return an absolute,
        canonical ``Path``. Raises ``ValueError``/``PermissionError`` on any
        escape attempt (unless the sandbox is disabled).
        """
        candidate = Path(path_str)
        if candidate.is_absolute():
            candidate = candidate.resolve()
        else:
            candidate = (self.root_dir / candidate).resolve()
        if self.enabled:
            self._assert_safe(candidate)
        return candidate

    # -- access checks ---------------------------------------------------

    def check_read(self) -> None:
        if not self.read_enabled:
            raise PermissionError("Read operations are disabled by sandbox policy.")

    def check_write(self) -> None:
        if not self.write_enabled:
            raise PermissionError("Write operations are disabled by sandbox policy.")

    def check_delete(self) -> None:
        if not self.delete_enabled:
            raise PermissionError("Delete operations are disabled by sandbox policy.")

    def resolve_mode(self, path: Path) -> Optional[str]:
        """Effective access mode for *path* under :attr:`path_modes`.

        Returns ``"read"`` / ``"read_write"`` / ``"blocked"``, or ``None`` when
        no rule covers the path (also treated as no access). Longest matching
        rule wins, so a per-file override beats its parent directory rule and a
        newly created file inherits the nearest granted directory. Returns
        ``None`` when no hierarchical policy is configured.
        """
        if self.path_modes is None:
            return None
        try:
            cand = path.resolve()
        except OSError:
            return None
        best_mode: Optional[str] = None
        best_len = -1
        for key, mode in self.path_modes.items():
            if self._is_under(cand, key):     # cand == key or under key
                klen = len(key.parts)
                if klen > best_len:
                    best_len = klen
                    best_mode = mode
        return best_mode

    def assert_writable(self, path: Path) -> None:
        """
        Path-dimension write check (assumes ``check_write`` was already called).
        Enforces the hierarchical policy or the ``writable_paths`` allowlist.
        """
        if not self.enabled:
            return
        if self.path_modes is not None:
            if self.resolve_mode(path) != "read_write":
                raise PermissionError(
                    f"Path '{path}' is read-only under the current sandbox "
                    f"policy (no read+write grant covers it)."
                )
            return
        if self.writable_paths is None:
            return
        if not any(self._is_under(path, w) for w in self.writable_paths):
            raise PermissionError(
                f"Path '{path}' is read-only under the current sandbox policy "
                f"(not in any writable directory)."
            )

    # -- concurrency -----------------------------------------------------

    def lock_paths(self, *paths: Path):
        """
        Context manager that serialises concurrent writes to *paths* (process-wide,
        keyed by resolved path) so parallel tool calls can't lose updates. Locks
        are acquired in canonical order to avoid deadlock on two-path operations.
        """
        from .file_locks import lock_paths as _lp
        return _lp(*paths)

    # -- visibility (non-raising) ---------------------------------------

    def is_allowed(self, path: Path) -> bool:
        """True if *path* is visible (readable under the policy)."""
        if not self.enabled:
            return True
        if self.path_modes is not None:
            return self.resolve_mode(path) in ("read", "read_write")
        try:
            candidate = path.resolve()
        except OSError:
            return False
        under_allowed = any(self._is_under(candidate, a) for a in self.allowed_paths)
        if not under_allowed:
            return False
        return not any(self._is_under(candidate, d) for d in self.denied_paths)

    def is_writable(self, path: Path) -> bool:
        """True if *path* may be written under the current policy."""
        if not self.write_enabled:
            return False
        if not self.enabled:
            return True
        if self.path_modes is not None:
            return self.resolve_mode(path) == "read_write"
        if self.writable_paths is None:
            return True
        return any(self._is_under(path.resolve(), w) for w in self.writable_paths)

    def filter_visible(self, paths: Iterable[Path]) -> list[Path]:
        """Drop any path that is not visible under the read policy."""
        return [p for p in paths if self.is_allowed(p)]

    # -- extension check -------------------------------------------------

    def check_extension(self, path: Path) -> None:
        """Raise ``PermissionError`` if the file extension is not allowed."""
        if self.allowed_extensions is not None:
            ext = path.suffix.lower()
            allowed_lower = {e.lower() for e in self.allowed_extensions}
            if ext and ext not in allowed_lower:
                raise PermissionError(
                    f"File extension '{ext}' is not allowed. "
                    f"Allowed: {', '.join(self.allowed_extensions)}"
                )

    # -- internal --------------------------------------------------------

    def _assert_safe(self, candidate: Path) -> None:
        """Raise if *candidate* escapes the sandbox."""
        if self.path_modes is not None:
            if self.resolve_mode(candidate) not in ("read", "read_write"):
                raise ValueError(
                    f"Path '{candidate}' is not covered by any granted "
                    f"permission (blocked or outside the selection)."
                )
            return
        under_allowed = any(
            self._is_under(candidate, allowed) for allowed in self.allowed_paths
        )
        if not under_allowed:
            raise ValueError(
                f"Path '{candidate}' is outside the allowed sandbox directories: "
                f"{', '.join(str(p) for p in self.allowed_paths)}"
            )
        for denied in self.denied_paths:
            if self._is_under(candidate, denied):
                raise PermissionError(
                    f"Path '{candidate}' is inside a denied directory: {denied}"
                )

    @staticmethod
    def _is_under(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    # -- summary for LLM -------------------------------------------------

    def describe_policy(self) -> str:
        """Return a human-readable policy summary for the system prompt."""
        if not self.enabled:
            return "Sandbox: DISABLED (no path confinement; full filesystem access subject to OS permissions)."
        if self.path_modes is not None:
            grants = "; ".join(
                f"{p} → {m}" for p, m in sorted(self.path_modes.items())
            ) or "(nothing granted)"
            lines = [
                f"Sandbox root: {self.root_dir}",
                f"Granted paths (subtrees inherit; new files included): {grants}",
                f"Read: {'enabled' if self.read_enabled else 'disabled'}",
                f"Write: {'enabled' if self.write_enabled else 'disabled'}",
                f"Max read size: {self.max_read_bytes} bytes",
                f"Max write size: {self.max_write_bytes} bytes",
            ]
            if self.allowed_extensions:
                lines.append(f"Allowed extensions: {', '.join(self.allowed_extensions)}")
            if self.dry_run:
                lines.append("Dry-run mode: ON (no real I/O performed)")
            return "\n".join(lines)
        lines = [
            f"Sandbox root: {self.root_dir}",
            f"Allowed paths: {', '.join(str(p) for p in self.allowed_paths)}",
            f"Read: {'enabled' if self.read_enabled else 'disabled'}",
            f"Write: {'enabled' if self.write_enabled else 'disabled'}",
            f"Delete: {'enabled' if self.delete_enabled else 'disabled'}",
            f"Max read size: {self.max_read_bytes} bytes",
            f"Max write size: {self.max_write_bytes} bytes",
        ]
        if self.denied_paths:
            lines.append(f"Denied paths: {', '.join(str(p) for p in self.denied_paths)}")
        if self.writable_paths is not None:
            lines.append(
                f"Writable paths: {', '.join(str(p) for p in self.writable_paths)} "
                f"(everything else is read-only)"
            )
        if self.allowed_extensions:
            lines.append(f"Allowed extensions: {', '.join(self.allowed_extensions)}")
        if self.dry_run:
            lines.append("Dry-run mode: ON (no real I/O performed)")
        return "\n".join(lines)
