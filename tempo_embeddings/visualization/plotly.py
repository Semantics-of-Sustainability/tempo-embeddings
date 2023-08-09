from typing import Iterable
import pandas as pd
import plotly.express as px
from dash import Dash
from dash import Input
from dash import Output
from dash import callback
from dash import dcc
from dash import html
from dash.html import Figure
from ..text.corpus import Corpus
from .visualizer import Visualizer


class PlotlyVisualizer(Visualizer):
    def __init__(self, *corpora: Iterable[Corpus]):
        self._corpora = corpora

    def _create_data(self, metadata_fields):
        data = pd.concat(
            (
                pd.concat(
                    [
                        pd.DataFrame(corpus.umap_embeddings(), columns=["x", "y"])
                        for corpus in self._corpora
                    ]
                ),
                pd.concat(
                    [
                        pd.DataFrame(corpus.hover_datas(metadata_fields))
                        for corpus in self._corpora
                    ]
                ),
            ),
            axis=1,
        )
        data.text = data.text.apply(PlotlyVisualizer._break_lines)
        data.year = data.year.astype(int)
        data["decade"] = (data.year / 10).astype(int) * 10

        return data

    def _create_scatter(self, data: pd.DataFrame, columns: list[str]) -> Figure:
        return px.scatter(
            data,
            x="x",
            y="y",
            color="corpus",
            size_max=60,
            hover_data={
                "x": False,
                "y": False,
            }
            | {column: True for column in columns},
            # animation_frame="decade",
            # animation_group="corpus",
        )

    @staticmethod
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

    def visualize(self, metadata_fields: list[str] = None):
        app = Dash(__name__)

        data = self._create_data(metadata_fields)

        hover_columns = ["text", "corpus"] + metadata_fields
        fig = self._create_scatter(data, columns=hover_columns)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        # TODO: remove axis labels

        div = [dcc.Graph(figure=fig, id="scatter")]

        if "year" in data.columns:
            div.append(
                dcc.Slider(
                    min=data.year.min(),
                    max=data.year.max(),
                    marks={str(year): str(year) for year in data.decade.unique()},
                    id="crossfilter-year--slider",
                )
            )
            div.append(html.Div(id="output-container-range-slider"))

            @callback(
                Output("scatter", "figure"),
                Input("crossfilter-year--slider", "value"),
            )
            def update_figure(selected_year):
                data = self._create_data(metadata_fields)

                if selected_year is not None:
                    decade = int(selected_year / 10) * 10
                    data = data[data.decade == decade]
                fig = self._create_scatter(data, columns=hover_columns)

                fig.update_layout(transition_duration=500)

                return fig

        app.layout = html.Div(div)

        app.run(debug=True)
