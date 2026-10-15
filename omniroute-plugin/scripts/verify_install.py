"""Show what Hermes loaded for this plugin in the active HERMES_HOME, using Hermes' own discovery
(the same path `hermes plugins doctor` uses), then optionally run one tool through Hermes' own
dispatcher.

    HERMES_HOME=/path/to/home python scripts/verify_install.py [tool_name [json_args]]

Run it with the Python environment Hermes is installed in.
"""

from __future__ import annotations

import json
import sys

from hermes_cli import plugins as hermes_plugins


def main() -> int:
    hermes_plugins.discover_plugins(force=True)
    manager = hermes_plugins.get_plugin_manager()
    loaded = manager._plugins.get("omniroute")
    if loaded is None:
        print("omniroute: not discovered (is it under $HERMES_HOME/plugins/ and in plugins.enabled?)")
        return 1
    print(f"plugin: {loaded.manifest.name} {loaded.manifest.version} enabled={loaded.enabled} "
          f"error={loaded.error!r}")
    from tools.registry import registry
    names = sorted(n for n in manager._plugin_tool_names
                   if (e := registry.get_entry(n, scope=manager.scope_key)) and e.toolset == "omniroute")
    for name in names:
        entry = registry.get_entry(name, scope=manager.scope_key)
        print(f"tool: {name:24s} toolset={entry.toolset} available={bool(entry.check_fn())}")
    hooks = sorted({h for h, cbs in manager._hooks.items() for cb in cbs
                    if "omniroute" in (getattr(cb, "__module__", "") or "")})
    print(f"hooks registered by omniroute: {hooks or 'none'}")
    for key, _label, _desc in hermes_plugins.get_plugin_toolsets():
        if key.startswith("omniroute"):
            members = sorted(n for n in names if registry.get_entry(n, scope=manager.scope_key).toolset == key)
            print(f"toolset: {key:16s} {members}")
    skill = manager.find_plugin_skill("omniroute:omniroute-routing")
    print(f"skill: omniroute:omniroute-routing -> {skill}")
    if len(sys.argv) > 1:
        import model_tools
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        raw = model_tools.handle_function_call(sys.argv[1], args, task_id="verify", session_id="verify")
        print(json.dumps(json.loads(raw), indent=2))
    return 0 if loaded.enabled and not loaded.error else 1


if __name__ == "__main__":
    sys.exit(main())
