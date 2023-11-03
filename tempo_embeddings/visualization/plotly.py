from typing import Iterable
from typing import Optional
import pandas as pd
import plotly.express as px
from dash import Dash
from dash import Input
from dash import Output
from dash import callback
from dash import dcc
from dash import html
from dash.html import Figure
from ..text.abstractcorpus import AbstractCorpus
from .visualizer import Visualizer


class PlotlyVisualizer(Visualizer):
    _SCATTER_ID: str = "scatter"
    _STANDARD_COLUMNS = ["text", "corpus"]
    _MARGIN_X = 0.5
    _MARGIN_Y = 0.5

    def __init__(self, *corpora: Iterable[AbstractCorpus]):
        self._corpora = corpora

    def _create_data(self, metadata_fields) -> pd.DataFrame:
        """Create a dataframe with the UMAP embeddings and metadata fields.

        Args:
            metadata_fields: The metadata fields to include in the dataframe.

        Returns:
            A dataframe with the UMAP embeddings and metadata fields from all corpora.
        """

        data = pd.concat(
            (
                pd.concat(
                    [
                        pd.DataFrame(corpus.embeddings, columns=["x", "y"])
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

    def _create_scatter(
        self,
        data: pd.DataFrame,
        columns: list[str],
        *,
        scale_x: Optional[tuple[float, float]] = None,
        scale_y: Optional[tuple[float, float]] = None,
    ) -> Figure:
        """Creates a scatter plot with the UMAP embeddings.

        Args:
            data: The dataframe with the UMAP embeddings and metadata fields.
            columns: The metadata fields to include in the hover data.
            scale_x: The range of the x-axis.
            scale_y: The range of the y-axis.

        Returns:
            A Figure object with the scatter plot.
        """
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
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        if scale_x is not None:
            fig.update_xaxes(range=scale_x)
        if scale_y is not None:
            fig.update_yaxes(range=scale_y)

        return fig

    def _add_slider(
        self,
        data: pd.DataFrame,
        metadata_fields: list[str],
        filter_field: str = "year",
        interval: int = 5,
    ) -> dcc.RangeSlider:
        """Generates a range slider for the a field.

        Args:
            data: The dataframe with the UMAP embeddings and metadata fields.
            metadata_fields: The metadata fields to include in the hover data.
            filter_field: The field to filter on. Needs to be numeric.
            interval: The interval between the marks on the slider. Defaults to 5.

        Returns:
            A RangeSlider object.
        """

        slider_id = f"crossfilter-{filter_field}--slider"
        start = data[filter_field].min()
        end = data[filter_field].max()
        marks = {
            str(value): str(value)
            for value in data[filter_field].unique()
            if value in (start, end) or value % interval == 0
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

            fig: Figure = self._create_scatter(
                data[data[filter_field].between(*selected_range)],
                columns=PlotlyVisualizer._STANDARD_COLUMNS + metadata_fields,
                scale_x=(
                    data.x.min() - PlotlyVisualizer._MARGIN_X,
                    data.x.max() + PlotlyVisualizer._MARGIN_X,
                ),
                scale_y=(
                    data.y.min() - PlotlyVisualizer._MARGIN_Y,
                    data.y.max() + PlotlyVisualizer._MARGIN_Y,
                ),
            )
            fig.update_layout()
            return fig

        return slider

    @staticmethod
    def _break_lines(
        text: str, max_line_length: int = 50, linebreak: str = "<br>"
    ) -> str:
        """Breaks a text into lines with a maximum length.

        Args:
            text: The text to break into lines.
            max_line_length: The maximum length of a line. Defaults to 50.

        Returns:
            The text with line breaks.
        """
        words = text.split()
        lines = []
        line = ""
        for word in words:
            if len(line) + len(word) > max_line_length:
                lines.append(line)
                line = ""
            line += f" {word}"
        lines.append(line)

        return linebreak.join(lines)

    def visualize(self, metadata_fields: Iterable[str] = None, jupyter_mode="external"):
        """Visualizes the UMAP embeddings.

        Args:
            metadata_fields: The metadata fields to include in the hover data.
        """
        app = Dash(__name__)

        metadata_fields = list(metadata_fields)

        data = self._create_data(metadata_fields)

        columns = PlotlyVisualizer._STANDARD_COLUMNS
        if metadata_fields:
            columns.extend(metadata_fields)

        fig = self._create_scatter(data, columns=columns)

        children = [dcc.Graph(figure=fig, id=PlotlyVisualizer._SCATTER_ID)]

        if "year" in data.columns:
            # Add range slider for filtering by year
            children.append(
                self._add_slider(data, metadata_fields, filter_field="year")
            )

        app.layout = html.Div(children)

        app.run(jupyter_mode=jupyter_mode)
