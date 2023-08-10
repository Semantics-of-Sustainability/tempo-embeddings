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
    _SCATTER_ID: str = "scatter"
    _STANDARD_COLUMNS = ["text", "corpus"]

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
        if "year" in data.columns:
            data.year = data.year.astype(int)

        return data

    def _create_scatter(self, data: pd.DataFrame, columns: list[str]) -> Figure:
        fig = px.scatter(
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
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        # TODO: remove axis labels

        return fig

    def _add_slider(
        self, data: pd.DataFrame, metadata_fields: list[str], filter_field: str = "year"
    ):
        slider_id = f"crossfilter-{filter_field}--slider"
        start = data[filter_field].min()
        end = data[filter_field].max()
        marks = {
            str(value): str(value)
            for value in data[filter_field].unique()
            if value in (start, end) or value % 5 == 0
        }

        slider = dcc.RangeSlider(
            min=start,
            max=end,
            step=1,
            marks=marks,
            id=slider_id,
            value=[start, end],
            tooltip={},
        )

        @callback(
            Output(PlotlyVisualizer._SCATTER_ID, "figure"), Input(slider_id, "value")
        )
        def update_figure(selected_range):
            # FIXME: do we need to re-create the dataframe here?
            data = self._create_data(metadata_fields)

            fig = self._create_scatter(
                data[data[filter_field].between(*selected_range)],
                columns=PlotlyVisualizer._STANDARD_COLUMNS + metadata_fields,
            )
            fig.update_layout(transition_duration=500)
            return fig

        return slider

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
        fig = self._create_scatter(
            data, columns=PlotlyVisualizer._STANDARD_COLUMNS + metadata_fields
        )

        div = [dcc.Graph(figure=fig, id=PlotlyVisualizer._SCATTER_ID)]

        if "year" in data.columns:
            # Add range slider for filtering by year
            div.append(self._add_slider(data, metadata_fields, filter_field="year"))

        app.layout = html.Div(div)

        app.run(debug=True)
