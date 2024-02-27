from threading import Thread
import fastapi
import numpy as np
import time
import uuid
from typing import List
import os
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch


PATH_INPUT = "/home/mcfrank/notes"
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 5020)

EXCLUDED_DIRS = [
    ".git",
    "/home/mcfrank/notes/.git",
    ".obsidian",
]

ALLOWED_EXTENSIONS = [
    ".md",
    ".txt",
]


app = fastapi.FastAPI()


class Link(BaseModel):
    source: str
    target: str


class Node(BaseModel):
    id: str
    name: str
    timestamp: str
    origin: str
    text: str


class Embedding(BaseModel):
    node_id: str
    embedding: List[float]


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
    c.execute(
        '''
            CREATE TABLE IF NOT EXISTS embeddings
            (node_id text, embedding text)
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


def embedding_exists(node_id) -> bool:
    # connet to data/sqlite.db
    conn = sqlite3.connect('data/sqlite.db')
    # check if a node exists in the database
    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM embeddings WHERE node_id = ?
        """,
        (node_id, )
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


def insert_embedding(embedding: Embedding):
    # connet to data/sqlite.db
    conn = sqlite3.connect('data/sqlite.db')
    # embedding list to string:
    embedding_str = embedding.embedding.__str__()
    print("embedding_str")
    print(embedding_str)
    # insert a node into the database
    c = conn.cursor()
    c.execute(
        """
            INSERT INTO embeddings (
                node_id, embedding
            ) VALUES (?, ?)
        """,
        (embedding.node_id, embedding_str)
    )
    conn.commit()


def files_to_nodes(directory):
    all_nodes = get_all_nodes()
    new_nodes = []
    for path, _, filenames in os.walk(directory):
        if any(excluded in path for excluded in EXCLUDED_DIRS):
            continue
        for filename in filenames:
            if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue
            id = str(uuid.uuid4())
            name = filename
            timestamp = str(int(time.time()*1000))
            origin = path + "/" + filename
            print(origin)
            with open(origin, "r") as f:
                text = f.read()
            # print(id, name, timestamp, origin, text)
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
                new_nodes.append(node)

    return all_nodes + new_nodes


def embed_text(text: str):
    # chunk the text into a column of lines of width 64 characters
    # handle the final lines with padding
    # i need to train this
    n_lines = 16
    padded_text = text.ljust(64 * (len(text) // 64 + 1))
    assert len(padded_text) % 64 == 0
    column_lines = [padded_text[i:i+64]
                    for i in range(0, len(padded_text), 64)]
    if len(column_lines) > n_lines:
        column_lines = column_lines[:16]

    embedded_chunks = embed_chunks(column_lines)

    principal_direction_abs = np.sum(embedded_chunks, axis=0)
    principal_direction_embedding = principal_direction_abs / \
        np.linalg.norm(principal_direction_abs)

    return principal_direction_embedding.tolist()


def embed_chunks(chunks: List[str]):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
        )

    # Sentences we want sentence embeddings for
    sentences = chunks

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.numpy()


def nodes_to_embeddings(nodes):
    embeddings = []
    print(f"attmpting to embed {len(nodes)} nodes")
    for node in nodes:
        if embedding_exists(node.id):
            print(f"embedding for node {node.id} already exists")
            embeddings.append(get_node_embedding(node.id))
            continue
        print(f"embedding node {node.id}")
        embedding_vector = embed_text(node.text)
        print(embedding_vector)
        embedding = Embedding(
            node_id=node.id,
            embedding=embedding_vector
        )
        print("inserting embedding")
        insert_embedding(embedding)
        embeddings.append(embedding)
    return embeddings


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


def get_all_embeddings() -> List[Embedding]:
    conn = sqlite3.connect('data/sqlite.db')

    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM embeddings
        """
    )
    temp_embeddings = []
    for row in c.fetchall():
        embedding_str = row[1]
        embedding_items_from_string = embedding_str.replace(
            "[", "").replace("]", "").split(",")
        embedding = [float(item) for item in embedding_items_from_string]
        temp_node = Embedding(
            node_id=row[0],
            embedding=embedding
        )
        temp_embeddings.append(temp_node)
    return temp_embeddings


def get_node_embedding(node_id) -> Embedding:
    conn = sqlite3.connect('data/sqlite.db')

    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM embeddings WHERE node_id = ?
        """,
        (node_id, )
    )
    row = c.fetchone()
    embedding_str = row[1]
    embedding_items_from_string = embedding_str.replace(
        "[", "").replace("]", "").split(",")
    embedding = [float(item) for item in embedding_items_from_string]

    return Embedding(
        node_id=row[0],
        embedding=embedding
    )


class EmbeddingRequest(BaseModel):
    node_id: str


class TextEmbeddingRequest(BaseModel):
    text: str


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return fastapi.responses.FileResponse("index.html")


@app.get("/nodes")
def get_nodes() -> List[Node]:
    snapshot_nodes = get_all_nodes()
    return snapshot_nodes


@app.post("/node/embedding")
def get_embedding(embedding_request: EmbeddingRequest) -> Embedding:
    node_id = embedding_request.node_id
    embedding = get_node_embedding(node_id)
    return embedding


@app.get("/embeddings")
def get_embeddings() -> List[Embedding]:
    return get_all_embeddings()


@app.post("/embed/text")
def embed_text_endpoint(
        text_embedding_request: TextEmbeddingRequest
) -> List[float]:
    return embed_text(text_embedding_request.text)


def start_embedding_process():
    node_embeddings = nodes_to_embeddings(nodes)
    print(f"embedded {len(node_embeddings)} nodes")


if __name__ == "__main__":
    import uvicorn
    create_root_node_table()
    nodes = files_to_nodes(PATH_INPUT)
    embedding_thread = Thread(target=start_embedding_process)
    embedding_thread.start()
    node_embeddings = get_all_embeddings()
    uvicorn.run(app, host=HOST, port=PORT)
