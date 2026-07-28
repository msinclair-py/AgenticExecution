#!/usr/bin/env python
"""Refresh this machine's Globus Compute tokens without an interactive stdin.

WHY THIS EXISTS
---------------
`Client()` triggers globus-sdk's *CommandLineLoginFlowManager*, which prints an
auth URL and then blocks on `input()` waiting for you to paste the code back. In
any non-interactive shell (an agent harness, a pipe, a `bash -c`) stdin is at EOF
and the flow dies with:

    CommandLineLoginFlowEOFError: An EOF was read when an authorization code was
    expected. (Are you running this in an interactive terminal?)

The fix is *LocalServerLoginFlowManager*: it starts a throwaway HTTP server on
localhost, opens the browser, and captures the authorization code straight from
the redirect. Nothing is typed, so no interactive terminal is required.

Also note: **logging in at globus.org in a browser does NOT refresh this
machine's token store.** They are separate credential caches. The local store
lives at ~/.globus_compute/storage.db (override with GLOBUS_COMPUTE_USER_DIR),
and a stale refresh token there surfaces as:

    AuthAPIError: ('POST', 'https://auth.globus.org/v2/oauth2/token',
                   None, 400, None, 'Bad Request')

Usage:
    python login.py          # refresh tokens, then list visible endpoints
    python login.py --check  # do NOT log in; just report token/endpoint state
"""
from __future__ import annotations

import argparse
import sys


def build_app():
    """A UserApp identical to the SDK's default, but with a browser-redirect flow."""
    from globus_sdk import GlobusAppConfig, UserApp
    from globus_sdk.login_flows import LocalServerLoginFlowManager

    # Reuse the SDK's own client id and token storage so the refreshed tokens
    # land where every other globus-compute process already looks for them.
    from globus_compute_sdk.sdk.auth.globus_app import (
        DEFAULT_CLIENT_ID,
        get_token_storage,
    )

    config = GlobusAppConfig(
        login_flow_manager=LocalServerLoginFlowManager,
        token_storage=get_token_storage(),
        request_refresh_tokens=True,  # long-lived; avoids re-login every session
    )
    return UserApp("globus-compute-mcp", client_id=DEFAULT_CLIENT_ID, config=config)


def report_endpoints(client) -> int:
    eps = client.get_endpoints()
    if not eps:
        print("\nNo endpoints are registered to this identity.")
        print("If you expected one, check you authenticated as the same Globus user")
        print("that owns the endpoint (`globus-compute-endpoint list` on the host).")
        return 0
    print(f"\n{len(eps)} endpoint(s) visible to this identity:\n")
    for e in eps:
        uuid, name = e.get("uuid"), e.get("name")
        try:
            st = client.get_endpoint_status(uuid)
            status = st.get("status", "?")
            workers = st.get("details", {}).get("total_workers")
            extra = f" workers={workers}" if workers is not None else ""
        except Exception as exc:  # status is best-effort; never fail the listing
            status, extra = f"<{type(exc).__name__}>", ""
        print(f"  {name or '<unnamed>':<36} {uuid}  status={status}{extra}")
    print("\nPut the UUID you want into skills/globus-compute-mcp/endpoints.yaml")
    return len(eps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report token/endpoint state without starting a login flow")
    args = ap.parse_args()

    from globus_compute_sdk import Client

    if args.check:
        try:
            report_endpoints(Client(do_version_check=False))
            print("\nTokens are VALID.")
            return 0
        except Exception as exc:
            print(f"Tokens are NOT usable: {type(exc).__name__}: {exc}")
            print("Run without --check to refresh them.")
            return 1

    print("Opening a browser for Globus authentication.")
    print("The auth code is captured via a localhost redirect - nothing to paste.\n")
    client = Client(app=build_app(), do_version_check=False)
    print("Authentication OK - tokens written to the globus-compute token store.")
    report_endpoints(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
