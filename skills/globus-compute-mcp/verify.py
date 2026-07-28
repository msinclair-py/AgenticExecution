#!/usr/bin/env python
"""Verification ladder for the local-MCP -> academy -> Globus Compute -> HPC stack.

Climb it in order. Every rung prints REAL VALUES, not just PASS/FAIL, because a
green tick that reports no number is how the previous campaign shipped a false
negative (see rung 8).

    # rungs 1-3 need NOTHING but the laptop -- no Globus, no endpoint, no SSH:
    ./.venv-gc311/bin/python skills/globus-compute-mcp/verify.py --max-rung 3

    # the full ladder, once the endpoints exist:
    ./.venv-gc311/bin/python skills/globus-compute-mcp/verify.py --endpoint aurora-gpu

Rungs
  1  local imports, versions, and the PYTHON VERSION PIN     (laptop only)
  2  Globus Auth token validity                              (laptop + network)
  3  --exchange local: the FULL MCP tool path in-process     (laptop only)
  4  endpoint_status + orchestrator/agent python match       (needs the endpoint)
  5  trivial call -> compute-node hostname                   (needs a PBS block)
  6  environment probe (numpy / MDAnalysis / OpenMM)
  7  accelerator pinning + visible device count
  8  concurrency: N tasks -> N distinct workers, wall ~ serial/N
  9  stateful session: an object survives between calls
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import server  # noqa: E402

# Every host in the stack must run this Python MINOR version. See rung 1.
REQUIRED_PY = (3, 11)

PASS, FAIL, SKIP, WARN = 'PASS', 'FAIL', 'SKIP', 'WARN'
_results: list[tuple[int, str, str, str]] = []


def record(rung: int, name: str, status: str, detail: str = '') -> None:
    _results.append((rung, name, status, detail))
    print(f'  [{status}] {name}' + (f' -- {detail}' if detail else ''))


def header(rung: int, title: str) -> None:
    print(f'\n=== RUNG {rung} -- {title} ' + '=' * max(4, 54 - len(title)))


# ---------------------------------------------------------------------------

def rung1() -> bool:
    header(1, 'local imports, versions, python pin')
    ok = True
    v = sys.version_info
    print(f'  python        {v.major}.{v.minor}.{v.micro}  ({sys.executable})')
    for mod in ('academy', 'globus_compute_sdk', 'fastmcp', 'dill', 'yaml'):
        try:
            m = __import__(mod)
            print(f'  {mod:22s} {getattr(m, "__version__", "present")}')
        except Exception as exc:                      # noqa: BLE001
            ok = False
            print(f'  {mod:22s} IMPORT FAILED: {type(exc).__name__}: {exc}')
    record(1, 'core imports', PASS if ok else FAIL)

    # THE version pin. Globus Compute serializes tasks and results with dill.
    # A minor-version skew between the orchestrator and an endpoint does NOT
    # raise a clear error -- it SEGFAULTS the interpreter. Verified locally:
    #   3.13-serialized blob deserialized under 3.11 -> exit 139, no traceback
    #   3.11-serialized blob deserialized under 3.13 -> exit 139, no traceback
    #   3.11 -> 3.11                                 -> works
    match = (v.major, v.minor) == REQUIRED_PY
    record(1, f'python is the pinned {REQUIRED_PY[0]}.{REQUIRED_PY[1]}',
           PASS if match else FAIL, f'{v.major}.{v.minor}.{v.micro}')
    if not match:
        print(f'  >>> EVERY host must run python {REQUIRED_PY[0]}.{REQUIRED_PY[1]}:')
        print('  >>> orchestrator, each Globus Compute endpoint, and every worker.')
        print('  >>> A skew SEGFAULTS (exit 139) with NO python traceback at all,')
        print('  >>> so it looks like a machine fault, not a version problem.')
        print(f'  >>>   uv venv --python {REQUIRED_PY[0]}.{REQUIRED_PY[1]} .venv-gc311')

    try:
        reg = server.load_registry()
        names = sorted(reg['endpoints'])
        conf = [n for n in names if reg['endpoints'][n].get('uuid')]
        print(f'  registry      {reg["path"]}')
        print(f'  endpoints     {names}')
        print(f'  configured    {conf or "NONE (bootstrap not done)"}')
        record(1, 'endpoint registry parses', PASS, f'{len(names)} entries')
    except Exception as exc:                          # noqa: BLE001
        ok = False
        record(1, 'endpoint registry parses', FAIL, f'{type(exc).__name__}: {exc}')

    shim = server._StdoutShim(sys.stderr.buffer)
    good = shim.buffer is sys.stderr.buffer and callable(shim.write)
    record(1, '_StdoutShim keeps .buffer as the real stream', PASS if good else FAIL)
    return ok and match and good


def rung2() -> bool:
    header(2, 'Globus Auth token validity')
    import os
    store = Path(os.environ.get('GLOBUS_COMPUTE_USER_DIR',
                                Path.home() / '.globus_compute')) / 'storage.db'
    if store.exists():
        age = (time.time() - store.stat().st_mtime) / 86400
        when = time.strftime('%Y-%m-%d', time.localtime(store.stat().st_mtime))
        print(f'  token store   {store}')
        print(f'                last written {when} ({age:.0f} days ago)')
    else:
        print(f'  token store   {store} -- DOES NOT EXIST (never logged in)')
    try:
        from globus_compute_sdk import Client
        eps = Client().get_endpoints()
        print(f'  live endpoints visible to this identity: {len(eps)}')
        for e in eps[:10]:
            print(f'    {e.get("uuid")}  {e.get("name")}')
        record(2, 'Globus Auth usable', PASS, f'{len(eps)} endpoint(s)')
        return True
    except Exception as exc:                          # noqa: BLE001
        record(2, 'Globus Auth usable', FAIL, f'{type(exc).__name__}: {exc}')
        print('\n  >>> RE-LOGIN REQUIRED. The previous campaign hit exactly this:')
        print('  >>> a token store last written 2025-07-09 that no longer refreshes,')
        print('  >>> reported as')
        print("  >>>   AuthAPIError('POST', 'https://auth.globus.org/v2/oauth2/token',")
        print("  >>>                None, 400, 'Error', 'Bad Request')")
        print('  >>> Fix (opens a browser):')
        print('  >>>   ./.venv-gc311/bin/python -c '
              '"from globus_compute_sdk import Client; Client()"')
        print('  >>> To keep the old store intact while testing, first:')
        print('  >>>   export GLOBUS_COMPUTE_USER_DIR=~/.globus_compute_selongatus')
        print('  >>> NOTE: rung 3 does not need this. Run --max-rung 3.')
        return False


# ---------------------------------------------------------------------------
# Rung 3 -- the whole MCP tool path, in-process, no Globus at all.
# ---------------------------------------------------------------------------

async def _rung3(endpoint: str | None, n: int, sleep_s: float) -> bool:
    from fastmcp import Client

    mcp, sess = server.build_server('local')
    ok = True
    async with Client(mcp) as c:
        tools = sorted(t.name for t in await c.list_tools())
        print(f'  tools         {tools}')
        expected = {'list_endpoints', 'endpoint_status', 'run_python',
                    'submit_batch', 'get_result', 'poll'}
        missing = expected - set(tools)
        record(3, 'all required tools exposed', PASS if not missing else FAIL,
               f'missing={sorted(missing)}' if missing else f'{len(tools)} tools')
        ok &= not missing

        le = (await c.call_tool('list_endpoints', {})).data
        live = [e['endpoint'] for e in le['endpoints'] if e['live']]
        print(f'  live agents   {live}')
        record(3, 'agents launched on every endpoint', PASS if live else FAIL,
               f'{len(live)} live')
        ok &= bool(live)
        ep = endpoint if endpoint in live else (live[0] if live else None)
        if ep is None:
            return False
        print(f'  using         {ep!r}')

        st = (await c.call_tool('endpoint_status', {'endpoint': ep})).data
        a = st['agent']
        print(f'  agent host    {a["hostname"]}  pid={a["pid"]}')
        print(f'  agent cwd     {a["cwd"]}   <- base_dir chdir')
        print(f'  agent python  {a["python_full"]}   orchestrator {st["orchestrator_python"]}')
        record(3, 'agent chdir()ed to its base_dir', PASS if a['cwd'] else FAIL,
               a['cwd'])

        r = (await c.call_tool('run_python',
                               {'code': 'print("hello from the agent")',
                                'endpoint': ep, 'wait': 60})).data
        print(f'  run_python    rc={r.get("returncode")} '
              f'stdout={r.get("stdout", "").strip()!r} mode={r.get("mode")}')
        good = r.get('returncode') == '0' and 'hello' in r.get('stdout', '')
        record(3, 'run_python (subprocess) returns stdout', PASS if good else FAIL)
        ok &= good

        r = (await c.call_tool('run_python',
                               {'code': 'raise ValueError("deliberate")',
                                'endpoint': ep, 'wait': 60})).data
        tb = r.get('stderr', '')
        good = r.get('returncode') not in (None, '0') and 'ValueError' in tb
        record(3, 'a failing snippet RETURNS rc!=0, does not raise',
               PASS if good else FAIL, f'rc={r.get("returncode")}')
        if good:
            print(f'  traceback     {tb.strip().splitlines()[-1]!r}')
        ok &= good

        r = (await c.call_tool('run_python',
                               {'code': 'import time; print("s", flush=True); time.sleep(30)',
                                'endpoint': ep, 'timeout': 3, 'wait': 60})).data
        good = r.get('returncode') == '-1' and 'Timed out' in r.get('stderr', '')
        record(3, 'worker-side timeout kills the subprocess',
               PASS if good else FAIL, r.get('stderr', '').strip()[:60])
        ok &= good

        # submit-then-poll: the call must return long before the work does
        t0 = time.time()
        r = (await c.call_tool('run_python',
                               {'code': 'import time; time.sleep(12); print("late")',
                                'endpoint': ep, 'timeout': 60, 'wait': 2})).data
        dt = time.time() - t0
        good = bool(r.get('pending')) and dt < 6
        record(3, 'wait elapsing returns pending+task_id (never blocks)',
               PASS if good else FAIL, f'returned in {dt:.1f}s, pending={r.get("pending")}')
        ok &= good
        late = r.get('task_id')

        # stateful persistence -- the reason to run agents at all
        await c.call_tool('run_python', {'code': 'EXPENSIVE = {"loaded": True}; '
                                                 'print("loaded once")',
                                         'endpoint': ep, 'stateful': True, 'wait': 60})
        r = (await c.call_tool('run_python',
                               {'code': 'print("still here:", EXPENSIVE)',
                                'endpoint': ep, 'stateful': True, 'wait': 60})).data
        good = 'still here' in r.get('stdout', '') and r.get('returncode') == '0'
        record(3, 'STATEFUL namespace survives between calls',
               PASS if good else FAIL, r.get('stdout', '').strip())
        ok &= good

        si = (await c.call_tool('session_info', {'endpoint': ep})).data
        print(f'  session holds {si["session_names"]}')
        record(3, 'session_info reports the retained object',
               PASS if 'EXPENSIVE' in si['session_names'] else FAIL,
               str(si['session_names']))

        # concurrency
        specs = [{'code': f'import time,os,socket\ntime.sleep({sleep_s})\n'
                          f'print("job {i}", os.getpid())', 'label': f'j{i}'}
                 for i in range(n)]
        t0 = time.time()
        b = (await c.call_tool('submit_batch',
                               {'specs': specs, 'endpoint': ep, 'timeout': 120})).data
        sub_dt = time.time() - t0
        print(f'  submit_batch  {b["count"]} tasks, returned in {sub_dt:.2f}s')
        record(3, 'submit_batch returns immediately (non-blocking)',
               PASS if sub_dt < 2.0 else FAIL, f'{sub_dt:.2f}s')

        s = (await c.call_tool('poll', {'task_ids': b['task_ids'],
                                        'summary_only': True})).data
        record(3, 'poll is immediate and reports pending',
               PASS if s['pending'] > 0 else WARN, f'pending={s["pending"]}')
        while True:
            s = (await c.call_tool('poll', {'task_ids': b['task_ids'],
                                            'summary_only': True})).data
            if s['pending'] == 0:
                break
            await asyncio.sleep(0.5)
        wall = time.time() - t0
        full = (await c.call_tool('poll', {'task_ids': b['task_ids']})).data
        pids = {r['pid'] for r in full['results'] if 'pid' in r}
        serial = n * sleep_s
        workers = sess.registry['endpoints'][ep].get('local_workers', 4)
        print(f'  wall          {wall:.1f}s  (serial would be {serial:.1f}s, '
              f'{workers} local workers)')
        print(f'  distinct pids {len(pids)}')
        print(f'  counts        {[(k, full[k]) for k in ("total","succeeded","failed_task","failed_infra")]}')
        good = full['succeeded'] == n and wall < serial * 0.75
        record(3, f'{n} tasks ran concurrently ({wall:.1f}s < {serial:.1f}s serial)',
               PASS if good else FAIL)
        ok &= good

        # The earlier task sleeps 12 s; poll it to completion rather than
        # assuming enough wall time has passed since (that assumption is what
        # made this check fail spuriously the first time).
        deadline = time.time() + 60
        while time.time() < deadline:
            g = (await c.call_tool('get_result', {'task_id': late})).data
            if not g.get('pending'):
                break
            await asyncio.sleep(0.5)
        good = g.get('returncode') == '0' and 'late' in g.get('stdout', '')
        record(3, 'get_result retrieves the earlier pending task',
               PASS if good else FAIL,
               f'rc={g.get("returncode")} stdout={g.get("stdout", "").strip()!r}')
        ok &= good

        # a batch with a deliberate failure -> classified as task, not infra
        b2 = (await c.call_tool('submit_batch',
                                {'specs': [{'code': 'print(1)', 'label': 'ok'},
                                           {'code': 'import sys; sys.exit(3)',
                                            'label': 'bad'}],
                                 'endpoint': ep, 'timeout': 30})).data
        while True:
            s = (await c.call_tool('poll', {'task_ids': b2['task_ids'],
                                            'summary_only': True})).data
            if s['pending'] == 0:
                break
            await asyncio.sleep(0.3)
        good = s['succeeded'] == 1 and s['failed_task'] == 1 and s['failed_infra'] == 0
        record(3, 'task failure vs infra failure classified correctly',
               PASS if good else FAIL,
               f'ok={s["succeeded"]} task_fail={s["failed_task"]} infra={s["failed_infra"]}')
        ok &= good
    return ok


def rung3(endpoint: str | None, n: int, sleep_s: float) -> bool:
    header(3, '--exchange local: full MCP tool path, no Globus')
    print('  Runs the orchestrator, the exchange and the agents in ONE process on')
    print('  thread pools. No Globus, no endpoint, no SSH. This exercises every')
    print('  tool, the academy Manager wiring, the handle rebinding, base_dir')
    print('  chdir, stateful sessions and submit-then-poll.')
    try:
        return asyncio.run(_rung3(endpoint, n, sleep_s))
    except Exception as exc:                          # noqa: BLE001
        import traceback
        traceback.print_exc()
        record(3, 'exchange-local end-to-end', FAIL, f'{type(exc).__name__}: {exc}')
        return False


# ---------------------------------------------------------------------------
# Rungs 4-9 -- need a live endpoint. Driven through the same MCP tools.
# ---------------------------------------------------------------------------

async def _remote_rungs(endpoint: str, max_rung: int, n: int, sleep_s: float,
                        wait: float) -> None:
    from fastmcp import Client

    mcp, sess = server.build_server('globus')
    async with Client(mcp) as c:
        # -- rung 4
        header(4, 'endpoint status + python version match')
        try:
            st = (await c.call_tool('endpoint_status', {'endpoint': endpoint})).data
        except Exception as exc:                      # noqa: BLE001
            record(4, 'endpoint reachable', FAIL, f'{type(exc).__name__}: {exc}')
            for r in range(5, 10):
                record(r, f'rung {r}', SKIP, 'rung 4 failed')
            return
        a = st['agent']
        print(f'  agent host    {a.get("hostname")}   pid={a.get("pid")}')
        print(f'  agent cwd     {a.get("cwd")}')
        print(f'  service       {st.get("service_status")} '
              f'{st.get("service_details", {})}')
        record(4, 'agent alive on the endpoint', PASS, str(a.get('hostname')))
        mism = 'PYTHON_VERSION_MISMATCH' in st
        print(f'  orchestrator  {st["orchestrator_python"]}')
        print(f'  agent python  {a.get("python_full")}')
        record(4, 'orchestrator and agent python versions match',
               FAIL if mism else PASS,
               st.get('PYTHON_VERSION_MISMATCH', 'matched'))
        if mism:
            print('  >>> STOP HERE. dill serialization across this skew SEGFAULTS')
            print('  >>> (exit 139, no traceback). Nothing below is trustworthy.')
            return

        # -- rung 5
        if max_rung >= 5:
            header(5, 'trivial call -> compute-node hostname')
            r = (await c.call_tool('run_python',
                                   {'code': 'import socket, os\n'
                                            'print("hostname", socket.gethostname())\n'
                                            'print("PBS_JOBID", os.environ.get("PBS_JOBID","<none>"))',
                                    'endpoint': endpoint, 'wait': wait})).data
            if r.get('pending'):
                record(5, 'trivial task completes', FAIL,
                       f'still pending after {wait}s -- PBS block queued? '
                       'OR a wrong engine.address.ifname (that HANGS SILENTLY)')
                return
            print(f'  stdout        {r.get("stdout", "").strip()!r}')
            record(5, 'trivial task returns a compute-node hostname',
                   PASS if r.get('returncode') == '0' else FAIL,
                   str(r.get('hostname')))

        # -- rung 6
        if max_rung >= 6:
            header(6, 'environment probe on the worker')
            si = (await c.call_tool('session_info', {'endpoint': endpoint})).data
            for k, v in (si.get('packages') or {}).items():
                if k != '_accelerators':
                    print(f'    {k:22s} {v}')
            r = (await c.call_tool('run_python',
                                   {'code': 'import os; print("TMPDIR", os.environ.get("TMPDIR"))',
                                    'endpoint': endpoint, 'wait': wait})).data
            tmp = r.get('TMPDIR', '')
            print(f'  worker TMPDIR {tmp!r}')
            record(6, 'worker TMPDIR is short (AF_UNIX ~108-char fix)',
                   PASS if tmp == '/tmp' else FAIL, f'TMPDIR={tmp!r}')

        # -- rung 7
        if max_rung >= 7:
            header(7, 'accelerator pinning')
            si = (await c.call_tool('session_info', {'endpoint': endpoint})).data
            acc = (si.get('packages') or {}).get('_accelerators', {})
            for k, v in acc.items():
                print(f'    {k:26s} {v}')
            pinned = bool(acc.get('ZE_AFFINITY_MASK') or acc.get('CUDA_VISIBLE_DEVICES'))
            expect = sess.registry['endpoints'][endpoint].get('accelerators_per_node')
            if expect:
                record(7, 'worker is pinned to one accelerator',
                       PASS if pinned else FAIL,
                       f'ZE={acc.get("ZE_AFFINITY_MASK")!r} '
                       f'CUDA={acc.get("CUDA_VISIBLE_DEVICES")!r}')
                print(f'  registry says {expect} accelerators/node -> that is the '
                      f'per-node concurrency')
            else:
                record(7, 'cpu endpoint: no accelerator expected',
                       PASS if not pinned else WARN)

        # -- rung 8
        if max_rung >= 8:
            header(8, f'concurrency -- {n} overlapping tasks, {sleep_s}s each')
            print('  Each probe SLEEPS. The previous campaign got a FALSE NEGATIVE')
            print('  here because the probe returned in microseconds -- every task')
            print('  finished before the next was dispatched, so 8 concurrent calls')
            print('  reported ONE worker on a perfectly healthy system. An')
            print('  instantaneous task cannot demonstrate parallelism.')
            specs = [{'code': f'import time, os, socket\ntime.sleep({sleep_s})\n'
                              f'print(socket.gethostname(), os.getpid(), '
                              f'os.environ.get("ZE_AFFINITY_MASK", '
                              f'os.environ.get("CUDA_VISIBLE_DEVICES","")))',
                      'label': f'c{i}'} for i in range(n)]
            t0 = time.time()
            b = (await c.call_tool('submit_batch',
                                   {'specs': specs, 'endpoint': endpoint,
                                    'timeout': sleep_s * 4 + 120})).data
            deadline = time.time() + wait
            while time.time() < deadline:
                s = (await c.call_tool('poll', {'task_ids': b['task_ids'],
                                                'summary_only': True})).data
                if s['pending'] == 0:
                    break
                await asyncio.sleep(2)
            wall = time.time() - t0
            full = (await c.call_tool('poll', {'task_ids': b['task_ids']})).data
            done = [r for r in full['results'] if not r.get('pending')]
            pids = {f'{r.get("hostname")}:{r.get("pid")}' for r in done}
            devs = {r.get('ZE_AFFINITY_MASK') or r.get('CUDA_VISIBLE_DEVICES')
                    or '<unpinned>' for r in done}
            serial = n * sleep_s
            print(f'  wall          {wall:.1f}s  (serial would be >= {serial:.1f}s)')
            print(f'  distinct workers  {len(pids)}  {sorted(pids)}')
            print(f'  distinct devices  {len(devs)}  {sorted(devs)}')
            record(8, 'every task succeeded',
                   PASS if full['succeeded'] == n else FAIL,
                   f'{full["succeeded"]}/{n}')
            record(8, f'{n} tasks landed on {n} distinct workers',
                   PASS if len(pids) >= n else FAIL, f'{len(pids)} distinct')
            record(8, 'no task landed on an UNPINNED worker',
                   PASS if '<unpinned>' not in devs else FAIL, str(sorted(devs)))
            record(8, f'wall {wall:.1f}s shows real overlap',
                   PASS if wall < serial * 0.75 else FAIL)

        # -- rung 9
        if max_rung >= 9:
            header(9, 'stateful session across calls')
            await c.call_tool('run_python',
                              {'code': 'import time\nT0 = time.time()\n'
                                       'BIG = list(range(1000000))\nprint("loaded", len(BIG))',
                               'endpoint': endpoint, 'stateful': True, 'wait': wait})
            r = (await c.call_tool('run_python',
                                   {'code': 'print("survived:", len(BIG))',
                                    'endpoint': endpoint, 'stateful': True,
                                    'wait': wait})).data
            print(f'  stdout        {r.get("stdout", "").strip()!r}')
            record(9, 'object loaded in one call is still resident in the next',
                   PASS if 'survived' in r.get('stdout', '') else FAIL,
                   r.get('stdout', '').strip())
            print('  This is why the agents are stateful: a Chai-1 model or an')
            print('  OpenMM Simulation loaded once is reused by every later call.')


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--endpoint', default=None,
                    help="endpoint name from endpoints.yaml (e.g. 'aurora-gpu')")
    ap.add_argument('--max-rung', type=int, default=9)
    ap.add_argument('--n', type=int, default=8, help='rungs 3/8: concurrent tasks')
    ap.add_argument('--sleep', type=float, default=5.0,
                    help='rungs 3/8: seconds each probe sleeps (MUST be > 0)')
    ap.add_argument('--wait', type=float, default=900.0,
                    help='seconds to wait for a remote rung (PBS queueing is slow)')
    args = ap.parse_args()

    if args.sleep <= 0:
        print('ERROR: --sleep must be > 0. A zero-length probe cannot demonstrate '
              'parallelism -- that is exactly the false negative rungs 3 and 8 '
              'exist to avoid.', file=sys.stderr)
        return 2

    print('Globus Compute MCP -- verification ladder')
    print(f'endpoint: {args.endpoint!r}   max rung: {args.max_rung}')

    ok1 = rung1()
    if args.max_rung >= 2:
        rung2()          # advisory: rung 3 does not need it
    else:
        record(2, 'Globus Auth', SKIP, '--max-rung < 2')

    if args.max_rung >= 3 and ok1:
        rung3(args.endpoint, args.n, args.sleep)
    elif args.max_rung >= 3:
        record(3, 'exchange-local end-to-end', SKIP, 'rung 1 failed; fix it first')

    if args.max_rung >= 4:
        if not ok1:
            for r in range(4, 10):
                record(r, f'rung {r}', SKIP, 'rung 1 failed')
        else:
            try:
                reg = server.load_registry()
                ep = args.endpoint or reg['default']
                rec = reg['endpoints'].get(ep, {})
                if not rec.get('uuid'):
                    for r in range(4, 10):
                        record(r, f'rung {r}', SKIP,
                               f'endpoint {ep!r} has no uuid -- one-time bootstrap '
                               'not done (ssh + globus-compute-endpoint start)')
                else:
                    asyncio.run(_remote_rungs(ep, args.max_rung, args.n,
                                              args.sleep, args.wait))
            except Exception as exc:                  # noqa: BLE001
                record(4, 'remote rungs', FAIL, f'{type(exc).__name__}: {exc}')

    print('\n=== SUMMARY ' + '=' * 54)
    worst = 0
    for rung, nm, status, detail in _results:
        print(f'  rung {rung}  {status:4s}  {nm}' + (f'  ({detail})' if detail else ''))
        if status == FAIL:
            worst = 1
    n_pass = sum(1 for r in _results if r[2] == PASS)
    n_fail = sum(1 for r in _results if r[2] == FAIL)
    n_skip = sum(1 for r in _results if r[2] == SKIP)
    print(f'\n  {n_pass} PASS   {n_fail} FAIL   {n_skip} SKIP\n')
    return worst


if __name__ == '__main__':
    raise SystemExit(main())
