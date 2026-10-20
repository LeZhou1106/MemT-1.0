"""MemT extensions; importing this package does not load the serving engine."""

from importlib import metadata

__version__ = "0.1.0"
SGLANG_VERSION = "0.5.16"
SGLANG_REFERENCE_COMMIT = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"


def require_sglang() -> None:
    """Check the serving dependency only when a server integration calls it."""
    try:
        installed_version = metadata.version("sglang")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"MemT serving requires sglang=={SGLANG_VERSION}; "
            "install memt-sglang[server] in the GPU environment."
        ) from error
    if installed_version != SGLANG_VERSION:
        raise RuntimeError(
            f"MemT serving requires sglang=={SGLANG_VERSION}; "
            f"found sglang=={installed_version}."
        )
