import abc
import logging
from typing import Optional

import pandas as pd
from bokeh.client import push_session
from bokeh.layouts import column
from bokeh.models import Legend
from bokeh.models.filters import GroupFilter
from bokeh.models.sources import CDSView, ColumnDataSource
from bokeh.models.widgets.sliders import RangeSlider
from bokeh.palettes import turbo
from bokeh.plotting import curdoc, figure, show
from bokeh.transform import factor_cmap

from ..settings import OUTLIERS_LABEL
from ..text.corpus import Corpus
from .visualizer import Visualizer


class BokehVisualizer(Visualizer):
    """Base class for visualizers using the Bokeh library."""

    _YEAR_COLUMN = "year"
    _TOP_WORDS_COLUMN = "top words"

    def __init__(self, *clusters: list[Corpus]):
        self._clusters = clusters

    def _select_palette(self):
        return turbo(len(self._clusters))

    @abc.abstractmethod
    def _create_data(self):
        return NotImplemented


class BokehInteractiveVisualizer(BokehVisualizer):
    """Interactive visualizer using the Bokeh library."""

    _LABEL_FIELD = "label"

    def __init__(
        self,
        *clusters: list[Corpus],
        metadata_fields: list[str] = None,
        legend_label_func: bool = lambda c: c.label,
        height: int = 500,
        width: int = 500,
    ):
        """Create an interactive Bokeh visualizer.

        Args:
            *clusters (list[Corpus]): Clusters to visualize.
            metadata_fields (list[str], optional): Metadata fields to show in the hover tooltip.
                If None (default), all metadata fields found in the data are shown.
            legend_label_func (bool, optional): Function to generate the legend label for a cluster. Defaults to using the cluster's label.
            height (int, optional): Height of the plot. Defaults to 500.
            width (int, optional): Width of the plot. Defaults to 500.
        """
        super().__init__(*clusters)

        self._metadata_fields = metadata_fields or list(
            {field for cluster in clusters for field in cluster.metadata_fields()}
        )
        self._init_figure(height, width)

        self._data: pd.DataFrame = self._create_data()
        self._source: ColumnDataSource = ColumnDataSource(self._data)
        self._legend_label_func = legend_label_func

    def _generate_tooltips(self, *, hover_width: str = "500px"):
        text_line: str = f"""
            <p style="width: {hover_width}; word-wrap: break-word">
                <b>Text:</b><br>
                @text
            </p>
            """
        metadata_lines: list[str] = [
            f"<b>{column}</b>: @{column}" for column in self._metadata_fields
        ]

        label_line: str = f"""
            <b>Corpus Label:</b> "@{self._LABEL_FIELD}"
            <br>
            <b>Corpus Top Words:</b> "@{self._TOP_WORDS_COLUMN}"
            """
        separator_line: str = "-" * 100

        return "<br>".join([text_line] + metadata_lines + [label_line, separator_line])

    def _init_figure(self, height: int, width: int):
        self._figure = figure(
            height=height, width=width, tooltips=self._generate_tooltips()
        )
        self._figure.add_layout(Legend(), "right")

    def _create_data(self) -> pd.DataFrame:
        return pd.concat(
            (self._create_cluster_data(cluster=cluster) for cluster in self._clusters)
        )

    def _create_cluster_data(self, *, cluster: Corpus) -> pd.DataFrame:
        hover_data: list[dict[str, str]] = pd.DataFrame(
            cluster.hover_datas(self._metadata_fields)
        )
        if self._YEAR_COLUMN in hover_data.columns:
            hover_data[self._YEAR_COLUMN] = hover_data[self._YEAR_COLUMN].astype(int)
        else:
            logging.warning(
                f"Column '{self._YEAR_COLUMN}' not found in cluster '{cluster}'."
            )

        for _column in self._metadata_fields:
            if _column not in hover_data.columns:
                logging.info(
                    "Column '%s' not found in cluster '%s'.", _column, cluster.label
                )

        return pd.concat((hover_data, cluster.coordinates()), axis="columns").assign(
            label=cluster.label
        )

    def _add_circles(self):
        palette = self._select_palette()
        labels = self._data[self._LABEL_FIELD].unique().tolist()

        for cluster in self._clusters:
            glyph = self._figure.circle(
                source=self._source,
                x="x",
                y="y",
                color=factor_cmap(self._LABEL_FIELD, palette, labels),
                legend_label=self._legend_label_func(cluster),
                view=CDSView(
                    filter=GroupFilter(
                        column_name=self._LABEL_FIELD, group=cluster.label
                    ),
                ),
            )

            if cluster.label == OUTLIERS_LABEL:
                glyph.muted = True

    def _year_slider(self) -> Optional[RangeSlider]:
        def callback(attr, old, new):  # noqa: unused-argument
            self._source.data = self._data.loc[
                self._data.year.between(new[0], new[1])
            ].to_dict(orient="list")

        if self._YEAR_COLUMN not in self._data.columns:
            logging.warning("No year data found. Skipping year slider.")
            return None

        min_year: int = self._data[self._YEAR_COLUMN].min()
        max_year: int = self._data[self._YEAR_COLUMN].max()

        slider = RangeSlider(
            start=min_year,
            end=max_year,
            value=(min_year, max_year),
            width=self._figure.frame_width,
        )
        slider.on_change("value_throttled", callback)
        return slider

    def _setup_legend(self):
        legend = self._figure.legend[0]

        ncols = 1 if legend.ncols == "auto" else legend.ncols
        while len(self._clusters) * legend.glyph_height / ncols > self._figure.height:
            ncols += 1
            logging.warning(
                "Legend heigt exceeds plot height (%d). Increasing number of columns to %d.",
                self._figure.height,
                ncols,
            )
        legend.ncols = ncols

        legend.click_policy = "hide"

    def _create_layout(self):
        self._add_circles()
        self._setup_legend()

        children = [self._figure]

        if slider := self._year_slider():
            children.append(slider)

        return column(*children)

    def create_document(self, doc):
        """Wrapper function for updating a document object.

        This is required when calling from within a notebook:

        visualizer = BokehInteractiveVisualizer(
            *clusters, metadata_fields=corpus.metadata_fields(), width=1200, height=1200
        )

        os.environ[
            "BOKEH_ALLOW_WS_ORIGIN"
        ] = "196g3qhrickgm9bpd0kgamlmid74eo61pes1eeu80dbm2djdbuos"

        show(visualizer.create_document)
        """

        doc.add_root(self._create_layout())

    def visualize(self):
        """Show an interactive plot.

        When calling from a notebook, use `show(visualizer.create_document)` instead.

        """
        # FIXME: this has not been tested; not required when calling from within a notebook.

        session = push_session(
            document=curdoc(), url="http://localhost:8050/"
        )  # TODO: add url and id

        layout = self._create_layout()
        session.document.add_root(self._create_layout())

        show(layout)

        session.loop_until_closed()
