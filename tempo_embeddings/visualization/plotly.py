from typing import Iterable
import pandas as pd
import plotly.express as px
from dash import Dash
from dash import dcc
from dash import html
from ..text.corpus import Corpus
from .visualizer import Visualizer


class PlotlyVisualizer(Visualizer):
    def __init__(self, *corpora: Iterable[Corpus]):
        self._corpora = corpora
        self._app = Dash(__name__)

    def visualize(self, metadata_fields: list[str] = None):
        def _break_lines(text: str, max_line_length: int = 50) -> str:
            words = text.split()
            lines = []
            line = ""
            for word in words:
                if len(line) + len(word) > max_line_length:
                    lines.append(line)
                    line = ""
                line += f" {word}"
            lines.append(line)
            return "<br>".join(lines)

        embeddings = pd.concat(
            [
                pd.DataFrame(corpus.umap_embeddings(), columns=["x", "y"])
                for corpus in self._corpora
            ]
        )

        hover_datas = pd.concat(
            [
                pd.DataFrame(corpus.hover_datas(metadata_fields))
                for corpus in self._corpora
            ]
        )
        hover_datas.text = hover_datas.text.apply(_break_lines)

        fig = px.scatter(
            pd.concat([embeddings, hover_datas], axis=1),
            x="x",
            y="y",
            # size="population",
            color="corpus",
            # hover_name="country",
            # log_x=True,
            size_max=60,
            hover_data={
                "x": False,
                "y": False,
            }
            | {column: True for column in hover_datas.columns},
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        self._app.layout = html.Div([dcc.Graph(figure=fig)])

        self._app.run(debug=True)
