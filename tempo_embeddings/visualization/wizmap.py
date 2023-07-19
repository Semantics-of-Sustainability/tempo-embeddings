"""Wrapper around Wizmap."""

import logging
import os
import socketserver
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from typing import Optional
import wizmap
from ..text.corpus import Corpus
from .visualizer import Visualizer


class WizmapVisualizer(Visualizer):
    data_file_name = "data.ndjson"
    grid_file_name = "grid.json"

    def __init__(
        self, corpus: Corpus, title: str, path: str = None, stopwords: list[str] = None
    ):
        super().__init__(corpus)

        self._path = path or tempfile.gettempdir()
        self._title = title

        self._data_write_args = {}
        if stopwords:
            logging.warning(
                "Pending on Wizmap PR: https://github.com/poloclub/wizmap/pull/11"
            )

        self._grid_write_args = {} if stopwords is None else {"stop_words": stopwords}

        self._server = None

    @property
    def url_prefix(self):
        address, port = self._server.server_address
        return f"http://{address}:{port}/" if self._server else None

    @property
    def data_file(self):
        path = os.path.join(self._path, self.data_file_name)
        return path if os.path.exists(path) else None

    @property
    def grid_file(self):
        path = os.path.join(self._path, self.grid_file_name)
        return path if os.path.exists(path) else None

    @property
    def data_url(self):
        return self.url_prefix + self.data_file_name

    @property
    def grid_url(self):
        return self.url_prefix + self.grid_file_name

    @property
    def stopwords(self) -> Optional[list[str]]:
        return self._grid_write_args.get("stop_words")

    def cleanup(self):
        self._stop_server()
        for file in (self.data_file, self.grid_file):
            if file:
                os.remove(file)

    def _write_data(self):
        """Write Wizmap visualizations to a file."""

        embeddings = self._corpus.umap_embeddings()
        xs = embeddings[:, 0].astype(float).tolist()
        ys = embeddings[:, 1].astype(float).tolist()

        data_list_args = self._data_write_args | {"xs": xs, "ys": ys}
        grid_list_args = self._grid_write_args | {"xs": xs, "ys": ys}

        metadata_fields = []
        if self._corpus.has_metadata("year"):
            metadata_fields.append("year")

            years = list(self._corpus.get_token_metadatas("year"))
            assert len(years) == len(xs)

            data_list_args["times"] = years
            grid_list_args["times"] = years
            grid_list_args["time_format"] = "%Y"

            # FIXME: this does not work for the default stopwords ("english")
            if self.stopwords is not None:
                self.stopwords.extend(years)
                self.stopwords.append("year")
                self.stopwords.append("br")

        texts = self._corpus.highlighted_texts(metadata_fields=metadata_fields)
        assert len(texts) == len(xs)
        data_list_args["texts"] = texts
        grid_list_args["texts"] = texts

        data_list = wizmap.generate_data_list(**data_list_args)
        grid_dict = wizmap.generate_grid_dict(**grid_list_args)

        wizmap.save_json_files(data_list, grid_dict, output_dir=self._path)

    def visualize(self, **kwargs):
        """Visualize the corpus with Wizmap."""
        if not (self.data_file and self.grid_file):
            self._write_data()
        if not self._server:
            self._serve(**kwargs)

        wizmap.visualize(self.data_url, self.grid_url)

    def _serve(self, **kwargs):
        if self._server is not None:
            raise RuntimeError("Server already running")

        port = kwargs.get("port", 8000)

        print("Starting server on port", port)
        handler = partial(self.WizmapRequestHandler, self)
        self._server = socketserver.TCPServer(("", port), handler)

        server_thread = threading.Thread(target=self._server.serve_forever)
        server_thread.start()

    def _stop_server(self):
        if self._server is None:
            logging.warning("Server not running.")
        else:
            self._server.server_close()
            self._server.shutdown()
            self._server = None

    def __del__(self):
        self.cleanup()


    class WizmapRequestHandler(SimpleHTTPRequestHandler):
        """A HTTP handler serving the Wizmap data files."""

        def __init__(self, visualizer, *args, **kwargs):
            for file in (visualizer.data_file, visualizer.grid_file):
                if not os.path.exists(file):
                    raise FileNotFoundError(file)

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
