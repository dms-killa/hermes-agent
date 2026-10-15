"""Serve the test fixtures as a strict mock of OmniRoute's management API, including the
stateful combo store (PUT /api/combos/{id} only; every other write is rejected).

    python scripts/serve_mock.py            # prints the URL and the placeholder key

For demos and for trying the plugin without a gateway. On SIGINT/SIGTERM it prints the request log
and any rejected requests (non-GET outside the combo route, paths outside the plugin's allowlist).
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location("hermes_omniroute", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["hermes_omniroute"] = package
spec.loader.exec_module(package)
from hermes_omniroute import endpoints  # noqa: E402
from mock_server import TEST_KEY, MockOmniRoute  # noqa: E402


def _stop(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    with MockOmniRoute(endpoints) as mock:
        print(f"OMNIROUTE_BASE_URL={mock.url}")
        print(f"OMNIROUTE_API_KEY={TEST_KEY}")
        sys.stdout.flush()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        print("request log:")
        for r in mock.requests:
            print(f"  {r['method']:4s} {r['path']}")
        print("violations:", json.dumps(mock.take_violations()))
