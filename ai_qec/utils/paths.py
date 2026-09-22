"""Locate the project checkout that owns ``datasets/`` and ``runs/``."""

from pathlib import Path

PROJECT_MARKER = Path("openspec") / "config.yaml"


def find_project_root(start: Path | str | None = None) -> Path:
    """Return the nearest ancestor of ``start`` (default: cwd) holding ``openspec/config.yaml``.

    If none is found, fall back to the source checkout of an editable install.
    A wheel installed into site-packages has no marker, so this raises instead of
    returning a directory where the runtime would write artifacts by mistake.
    """

    origin = Path.cwd() if start is None else Path(start)
    origin = origin.resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / PROJECT_MARKER).is_file():
            return candidate
    source_checkout = Path(__file__).resolve().parents[2]
    if (source_checkout / PROJECT_MARKER).is_file():
        return source_checkout
    raise FileNotFoundError(
        f"no directory containing {PROJECT_MARKER.as_posix()} above {origin}, and ai_qec is not "
        "an editable install of the project; pass the project root explicitly"
    )
