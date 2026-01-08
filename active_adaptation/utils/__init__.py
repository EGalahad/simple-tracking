from typing import Any, Mapping, Iterable, Sized


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: An instance of a class to update.
        data: Input dictionary to update from.
        _ns: Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():
        # key_ns is the full namespace of the key
        key_ns = _ns + "/" + key

        # -- A) if key is present in the object ------------------------------------
        if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)

            # -- 1) nested mapping → recurse ---------------------------
            if isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue

            # -- 2) iterable (list / tuple / etc.) ---------------------
            if isinstance(value, Iterable) and not isinstance(value, str):
                # ---- 2a) flat iterable → replace wholesale ----------
                if all(not isinstance(el, Mapping) for el in value):
                    out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                    if isinstance(obj, dict):
                        obj[key] = out_val
                    else:
                        setattr(obj, key, out_val)
                    continue

                # ---- 2b) existing value is None → abort -------------
                if obj_mem is None:
                    raise ValueError(
                        f"[Config]: Cannot merge list under namespace: {key_ns} because the existing value is None."
                    )

                # ---- 2c) length mismatch → abort -------------------
                if (
                    isinstance(obj_mem, Sized)
                    and isinstance(value, Sized)
                    and len(obj_mem) != len(value)
                ):
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )

                # ---- 2d) keep tuple/list parity & recurse ----------
                if isinstance(obj_mem, tuple):
                    value = tuple(value)
                else:
                    set_obj = True
                    # recursively call if iterable contains Mappings
                    for i in range(len(obj_mem)):
                        if isinstance(value[i], Mapping):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False
                    # do not set value to obj, otherwise it overwrites the cfg class with the dict
                    if not set_obj:
                        continue

            # -- 3) callable attribute → resolve string --------------
            elif callable(obj_mem):
                # update function name
                value = string_to_callable(value)

            # -- 4) simple scalar / explicit None ---------------------
            elif value is None or isinstance(value, type(obj_mem)):
                pass

            # -- 5) type mismatch → abort -----------------------------
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )

            # -- 6) final assignment ---------------------------------
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)

        # -- B) if key is not present ------------------------------------
        else:
            if isinstance(value, Mapping):
                raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")
            if isinstance(value, Iterable) and not isinstance(value, str):
                raise KeyError(
                    f"[Config]: Key not found under namespace: {key_ns}. Expected a list or tuple."
                )
            if callable(value):
                raise KeyError(
                    f"[Config]: Key not found under namespace: {key_ns}. Expected a callable."
                )

            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)
