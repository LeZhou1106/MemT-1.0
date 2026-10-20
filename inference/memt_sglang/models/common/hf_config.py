"""Identify the MemT release and compare fixed checkpoint fields."""

from collections.abc import Mapping

# Published config.json model_type of each released model. The value is the
# name used everywhere outside the backbone-specific packages (models/qwen,
# models/deepseek): "memt-nano" is MemT-1.0-Nano, "memt-flash" is MemT-1.0-Flash.
MODEL_TYPES = {
    "memt_qwen3_5_moe": "memt-nano",
    "memt_deepseek_v4": "memt-flash",
}


def memt_model(config):
    """Return "memt-nano" or "memt-flash" for a published MemT-1.0 config, else None."""
    fields = config if isinstance(config, Mapping) else config.to_dict()
    return MODEL_TYPES.get(fields.get("model_type"))


def validate_fixed_fields(
    fields, fixed_fields, *, section, exact=False, variable_fields=()
):
    """Check pinned values; variable_fields must be present and are read as-is.

    Pinned fields select code paths that the released kernels implement only
    once. With exact=True, any other field is rejected, so a config with a new
    switch fails here instead of being served with the wrong algorithm.
    """
    if not isinstance(fields, Mapping):
        raise ValueError(f"{section} must be an object")
    known_fields = fixed_fields.keys() | set(variable_fields)
    # Outside exact mode an absent pinned field reads as None, e.g. an
    # unquantized checkpoint without quantization_config.
    required_fields = known_fields if exact else set(variable_fields)
    missing_fields = sorted(required_fields - fields.keys())
    unknown_fields = sorted(fields.keys() - known_fields) if exact else []
    if missing_fields or unknown_fields:
        raise ValueError(
            f"Invalid {section} fields: missing={missing_fields}, "
            f"unknown={unknown_fields}"
        )
    for field_name, expected_value in fixed_fields.items():
        actual_value = fields.get(field_name)
        field_path = f"{section}.{field_name}"
        if isinstance(expected_value, Mapping):
            validate_fixed_fields(
                actual_value, expected_value, section=field_path, exact=True
            )
        elif not _same_value(actual_value, expected_value):
            raise ValueError(f"Unsupported {field_path}: expected {expected_value!r}")


def _same_value(actual_value, expected_value):
    if type(actual_value) is not type(expected_value):
        return False
    if isinstance(expected_value, list):
        return len(actual_value) == len(expected_value) and all(
            _same_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual_value, expected_value)
        )
    return actual_value == expected_value
