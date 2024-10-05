import lancedb
from lancedb.pydantic import LanceModel, Vector
from .config import get_project_root


def lancedb_setup(ndims: int):

    class DataModel(LanceModel):
        vector: Vector(dim=ndims)  # type: ignore
        text: str
        source: str
        page: str
        id: str

    uri = get_project_root() + "/db"
    db = lancedb.connect(uri)
    table = db.create_table("rag", schema=DataModel, mode="overwrite")
    return table


def lancedb_table():

    uri = get_project_root() + "/db"
    db = lancedb.connect(uri)
    table = db.open_table("rag")
    return table
