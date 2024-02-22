import fastapi
import time
import uuid
from typing import List
import json
import os
import random
import networkx as nx
from networkx.readwrite import json_graph
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3


PATH_INPUT = "/home/mcfrank/brain/data/test"

app = fastapi.FastAPI()

graph = nx.Graph()


class Link(BaseModel):
    source: str
    target: str


class Node(BaseModel):
    id: str
    name: str
    timestamp: str
    origin: str
    text: str


def create_root_node_table():
    # connet to data/sqlite.db
    conn = sqlite3.connect('data/sqlite.db')
    # creates a table in the database if not exists
    c = conn.cursor()
    c.execute(
        '''
            CREATE TABLE IF NOT EXISTS root_nodes
            (id text, name text, timestamp text, origin text, text text)
        '''
    )
    conn.commit()


def root_node_exists(absolute_path) -> bool:
    # connet to data/sqlite.db
    conn = sqlite3.connect('data/sqlite.db')
    # check if a node exists in the database
    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM root_nodes WHERE origin = ?
        """,
        (absolute_path, )
    )

    return c.fetchone() is not None


def insert_root_node(node: Node):
    # connet to data/sqlite.db
    conn = sqlite3.connect('data/sqlite.db')
    # insert a node into the database
    c = conn.cursor()
    c.execute(
        """
            INSERT INTO root_nodes (
                id, name, timestamp, origin, text
            ) VALUES (?, ?, ?, ?, ?)
        """,
        (node.id, node.name, node.timestamp, node.origin, node.text)
    )
    conn.commit()


def files_to_nodes(directory):
    nodes = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            id = str(uuid.uuid4())
            name = filename
            timestamp = str(int(time.time()*1000))
            origin = path + "/" + filename
            text = open(origin).read()
            print(id, name, timestamp, origin, text)
            node = Node(
                id=id,
                name=name,
                timestamp=timestamp,
                origin=origin,
                text=text
            )
            node_already_in_db = root_node_exists(origin)

            print("-"*16)
            if not node_already_in_db:
                print(f"inserting node {node}")
                insert_root_node(node)
            else:
                print(f"node {node} already exists thus not inserting")
            print("-"*16)
            nodes.append(node)

    return nodes


def get_all_nodes() -> List[Node]:
    conn = sqlite3.connect('data/sqlite.db')

    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM root_nodes
        """
    )
    return [
        Node(
            id=row[0],
            name=row[1],
            timestamp=row[2],
            origin=row[3],
            text=row[4]
        )
        for row in c.fetchall()
    ]


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return fastapi.responses.FileResponse("index.html")


@app.get("/nodes")
def get_nodes() -> List[Node]:
    snapshot_nodes = get_all_nodes()
    return snapshot_nodes


if __name__ == "__main__":
    import uvicorn
    create_root_node_table()
    nodes = files_to_nodes("/home/mcfrank/brain/data/test")
    print("-" * 16 + "gathered these nodes")
    print(len(nodes))
    print("-" * 16)
    uvicorn.run(app, host="localhost", port=5020)
