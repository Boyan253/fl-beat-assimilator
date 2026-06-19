"""Make PyFLP 2.2.1 work on Python 3.11+.

PyFLP's base `EventEnum` is intentionally empty and dispatches IDs to its
subclasses via `_missing_`. Python 3.11 forbids *calling* an empty enum
(`TypeError: ... has no members defined`), which breaks every `EventEnum(id)`
call in the parser. We patch the enum metaclass so calling the empty base
routes through `_missing_` (subclass dispatch + pseudo-members) as before.

    import pyflp_patch  # noqa: F401  (apply before pyflp.parse)
    import pyflp
"""
import pyflp._events as _ev

_real = _ev.EventEnum
_meta = type(_real)
if not getattr(_meta, "_flmidi_patched", False):
    _orig_call = _meta.__call__

    def _patched_call(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs and not cls._member_map_:
            r = cls._missing_(args[0])
            if r is not None:
                return r
        return _orig_call(cls, *args, **kwargs)

    _meta.__call__ = _patched_call
    _meta._flmidi_patched = True
