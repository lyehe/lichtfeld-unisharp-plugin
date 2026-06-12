from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from core import repo_policy


def _install_lf_stub(monkeypatch):
    class _Panel:
        pass

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=SimpleNamespace(MAIN_PANEL_TAB="MAIN_PANEL_TAB"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        free_plugin_textures=lambda _plugin_name: None,
    )
    lf_stub.log = SimpleNamespace(
        warn=lambda _msg: None,
        info=lambda _msg: None,
        error=lambda _msg: None,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


def _import_panel_module(monkeypatch):
    root = Path(__file__).resolve().parent.parent
    package_name = repo_policy.PLUGIN_LINK_NAME
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = ModuleType(package_name)
    package.__file__ = str(root / "__init__.py")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package_name, package)
    return importlib.import_module(f"{package_name}.panels.main_panel")


def test_panel_imports_and_draw_is_noop(monkeypatch):
    _install_lf_stub(monkeypatch)
    main_panel = _import_panel_module(monkeypatch)
    panel = main_panel.UnisharpPanel()
    panel.draw(SimpleNamespace())


def test_panel_bind_model_registers_without_missing_methods(monkeypatch):
    _install_lf_stub(monkeypatch)
    main_panel = _import_panel_module(monkeypatch)
    panel = main_panel.UnisharpPanel()

    class _Model:
        def bind(self, *_args, **_kwargs):
            pass

        def bind_func(self, *_args, **_kwargs):
            pass

        def bind_event(self, *_args, **_kwargs):
            pass

        def get_handle(self):
            return SimpleNamespace(dirty=lambda _name: None)

    ctx = SimpleNamespace(create_data_model=lambda _name: _Model())
    panel.on_bind_model(ctx)
    assert panel._handle is not None
