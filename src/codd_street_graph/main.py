import asyncio
from pathlib import Path
import tempfile
import zipfile
from uuid import uuid4

from .models import StreetGraphPair
from .config import (
    default_houses_filename,
    default_streets_filename,
    default_stations_filename,
)
from .ml import process
from .util import serialize


def count_flow(zip_path: Path) -> StreetGraphPair:
    temp_dir = Path(tempfile.gettempdir()) / str(uuid4())
    temp_dir.mkdir()

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(temp_dir)

    new_graph, old_graph, houses = process(
        temp_dir / default_houses_filename,
        temp_dir / default_streets_filename,
        temp_dir / default_stations_filename,
    )

    return serialize(new_graph, old_graph, houses)
