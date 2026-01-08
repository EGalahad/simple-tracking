import inspect


class RegistryMixin:
    def __init_subclass__(cls) -> None:
        """Put the subclass in the global registry."""
        if not hasattr(cls, "registry"):
            cls.registry = {}

        cls_name = cls.__name__
        try:
            cls._file = inspect.getfile(cls)
            cls._line = inspect.getsourcelines(cls)[1]
        except Exception:
            cls._file = "unknown"
            cls._line = "unknown"

        if cls_name.startswith("_"):
            return
        if cls_name not in cls.registry:
            cls.registry[cls_name] = cls
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")
