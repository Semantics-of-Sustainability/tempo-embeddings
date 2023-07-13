"""Wrapper around Wizmap."""

import logging
import os
import socketserver
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
import wizmap
from ..text.corpus import Corpus
from .visualizer import Visualizer


class WizmapVisualizer(Visualizer):
    data_file_name = "data.ndjson"
    grid_file_name = "grid.json"

    def __init__(self, corpus: Corpus, title: str, path: str = None):
        super().__init__(corpus)
        self._path = path or tempfile.gettempdir()
        self._title = title

        self._server = None

    @property
    def url_prefix(self):
        address, port = self._server.server_address
        return f"http://{address}:{port}/" if self._server else None

    @property
    def data_file(self):
        return os.path.join(self._path, self.data_file_name)

    @property
    def grid_file(self):
        return os.path.join(self._path, self.grid_file_name)

    @property
    def data_url(self):
        return self.url_prefix + self.data_file_name

    @property
    def grid_url(self):
        return self.url_prefix + self.grid_file_name

    def write_data(self):
        """Write Wizmap visualizations to a file."""

        embeddings = self._corpus.umap_embeddings()
        xs = embeddings[:, 0].astype(float).tolist()
        ys = embeddings[:, 1].astype(float).tolist()
        texts = self._corpus.highlighted_texts()

        data_list = wizmap.generate_data_list(xs, ys, texts)
        grid_dict = wizmap.generate_grid_dict(xs, ys, texts, self._title)

        wizmap.save_json_files(data_list, grid_dict, output_dir=self._path)

    def visualize(self, **kwargs):
        """Visualize the corpus with Wizmap."""
        # self.write_data()

        # self.serve(**kwargs)

        wizmap.visualize(self.data_url, self.grid_url)

    def serve(self, **kwargs):
        if self._server is not None:
            raise RuntimeError("Server already running")

        port = kwargs.get("port", 8000)

        print("Starting server on port", port)
        handler = partial(WizmapRequestHandler, self)
        self._server = socketserver.TCPServer(("", port), handler)

        server_thread = threading.Thread(target=self._server.serve_forever)
        server_thread.start()

    def stop_server(self):
        if self._server is None:
            logging.warning("Server not running")
        else:
            self._server.server_close()
            self._server.shutdown()
            self._server = None


class WizmapRequestHandler(SimpleHTTPRequestHandler):
    """A HTTP handler serving the Wizmap data files."""

    def __init__(self, visualizer, *args, **kwargs):
        if not os.path.exists(visualizer.data_file):
            raise FileNotFoundError(visualizer.data_file)
        if not os.path.exists(visualizer.grid_file):
            raise FileNotFoundError(visualizer.grid_file)
        self._visualizer = visualizer

        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" + self._visualizer.data_file_name:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(self._visualizer.data_file, "rb") as file:
                self.wfile.write(file.read())
        elif self.path == "/" + self._visualizer.grid_file_name:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(self._visualizer.grid_file, "rb") as file:
                self.wfile.write(file.read())
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"File not found")
