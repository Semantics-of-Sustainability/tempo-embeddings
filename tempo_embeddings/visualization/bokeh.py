import abc
import logging
import pandas as pd
from bokeh.client import push_session
from bokeh.layouts import column
from bokeh.models import Legend
from bokeh.models.filters import GroupFilter
from bokeh.models.sources import CDSView
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.sliders import RangeSlider
from bokeh.palettes import turbo
from bokeh.plotting import curdoc
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.transform import factor_cmap
from ..settings import OUTLIERS_LABEL
from ..text.abstractcorpus import AbstractCorpus
from .visualizer import Visualizer


class BokehVisualizer(Visualizer):
    """Base class for visualizers using the Bokeh library."""

    _YEAR_COLUMN = "year"

    def __init__(self, *clusters: list[AbstractCorpus]):
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
        *clusters: list[AbstractCorpus],
        metadata_fields: list[str] = None,
        height: int = 500,
        width: int = 500,
    ):
        """Create an interactive Bokeh visualizer.

        Args:
            *clusters (list[AbstractCorpus]): Clusters to visualize.
            metadata_fields (list[str], optional): Metadata fields to show in the hover tooltip.
                If None (default), all metadata fields found in the data are shown.
            height (int, optional): Height of the plot. Defaults to 500.
            width (int, optional): Width of the plot. Defaults to 500.
        """
        super().__init__(*clusters)

        self._metadata_fields = metadata_fields or list(
            {cluster.metadata_fields() for cluster in clusters}
        )
        self._init_figure(height, width)

        self._data: pd.DataFrame = self._create_data()
        self._source: ColumnDataSource = ColumnDataSource(self._data)

    def _init_figure(self, height: int, width: int):
        tool_tips = [
            ("corpus", f"@{self._LABEL_FIELD}"),
            ("text", "@text"),
        ] + [(column, "@" + column) for column in self._metadata_fields]

        self._figure = figure(height=height, width=width, tooltips=tool_tips)

        self._figure.add_layout(Legend(), "right")

    def _create_data(self) -> pd.DataFrame:
        return pd.concat(
            (self._create_cluster_data(cluster=cluster) for cluster in self._clusters)
        )

    def _create_cluster_data(self, *, cluster: AbstractCorpus) -> pd.DataFrame:
        hover_data: list[dict[str, str]] = pd.DataFrame(
            cluster.hover_datas(self._metadata_fields)
        )

        for _column in self._metadata_fields:
            if _column not in hover_data.columns:
                logging.info(
                    "Column '%s' not found in cluster '%s'.", _column, cluster.label
                )

        return pd.concat(
            (hover_data.astype({self._YEAR_COLUMN: int}), cluster.embeddings_as_df()),
            axis="columns",
        ).assign(label=cluster.label)

    def _add_circles(self):
        palette = self._select_palette()
        labels = self._data[self._LABEL_FIELD].unique().tolist()

        for cluster in self._clusters:
            glyph = self._figure.circle(
                source=self._source,
                x="x",
                y="y",
                color=factor_cmap(self._LABEL_FIELD, palette, labels),
                legend_label=cluster.label,
                view=CDSView(
                    filter=GroupFilter(
                        column_name=self._LABEL_FIELD, group=cluster.label
                    ),
                ),
            )

            if cluster.label == OUTLIERS_LABEL:
                glyph.visible = False

    def _year_slider(self) -> RangeSlider:
        def callback(attr, old, new):  # noqa: unused-argument
            self._source.data = self._data.loc[
                self._data.year.between(new[0], new[1])
            ].to_dict(orient="list")

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

        # Does 'auto' always resolve to 1 column?
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
        slider = self._year_slider()

        return column(self._figure, slider)

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
        # FIXME: this has not been tested; not required when calling from within a .

        session = push_session(
            document=curdoc(), url="http://localhost:8050/"
        )  # TODO: add url and id

        layout = self._create_layout()
        session.document.add_root(self._create_layout())

        show(layout)

        session.loop_until_closed()
