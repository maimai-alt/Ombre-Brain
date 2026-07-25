# Test isolation

Ombre Brain tests must not be started with bare `pytest` or `python -m pytest`.
The only supported entry point is `scripts/run_isolated_tests.py`, which creates
a disposable root and starts pytest as a child process with a minimal,
allowlisted environment.

Install the Python 3.12 development lock without resolving new versions:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock.txt
```

Run the default offline suite:

```powershell
.\.venv\Scripts\python.exe scripts/run_isolated_tests.py -m "not external" tests -q
```

Arguments after the script name are parsed by a deny-by-default allowlist. Test
targets must stay below `tests/`; supported selection and display options include
`-q`, `-v`, `-vv`, `-k`, `-m`, `--maxfail`, `--tb`, `--color`, and `--capture`.
The launcher rejects plugin, config, rootdir, collection-boundary, external,
cache, and temporary-directory overrides. It always combines marker selection
with `not external`, then adds the isolated `--basetemp`, `cache_dir`, and the
explicit `pytest-asyncio` / `pytest-timeout` plugins itself.

The child receives only a small set of Windows startup and executable-discovery
variables. HOME, TEMP, Python bytecode, pytest cache, model caches,
Rust homes, Ombre configuration, logs, buckets, embeddings, archive, and outbox
all resolve below the disposable root. Reads and writes that would normally use
the repository `src/.env` are redirected to a per-test file below that root.
API keys, proxies, Docker credentials, SSH agents, external URLs, and real
Ombre configuration are never forwarded.

Tests marked `external` are skipped unless explicitly selected, and the standard
launcher still does not import credentials from its parent. Real integration
validation requires a separately reviewed, purpose-built environment; it is not
part of the default pytest workflow.

The launcher removes its root after success, test failure, ordinary launcher
exceptions, and KeyboardInterrupt. An uncatchable process termination can leave
an `ombre-pytest-*` directory in the system temporary directory. Cleanup code
requires process-local ownership plus that exact prefix directly below the
system temp directory, and must never scan user data locations.
