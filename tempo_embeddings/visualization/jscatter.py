import logging
from typing import Optional

import jscatter
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import widgets

from ..settings import STOPWORDS
from ..text.corpus import Corpus
from ..text.keyword_extractor import KeywordExtractor


class JScatterVisualizer:
    """A class for creating interactive scatter plots with Jupyter widgets."""

    def __init__(
        self,
        corpus,
        categorical_fields: list[str] = ["newspaper", "label"],
        continuous_filter_fields: list[str] = ["year"],
        tooltip_fields: list[str] = ["year", "text", "label", "top words", "newspaper"],
        fillna: dict[str, str] = {"newspaper": "NRC"},
        color_by: str = "label",
        keyword_extractor: Optional[KeywordExtractor] = None,
    ):
        self._keyword_extractor = keyword_extractor or KeywordExtractor(
            corpus, exclude_words=STOPWORDS
        )
        self._categorical_fields = categorical_fields
        self._continuous_filter_fields = continuous_filter_fields
        self._tooltip_fields = tooltip_fields
        self._fillna = fillna
        self._color_by = color_by

        self._plot = PlotWidgets(
            [corpus],
            self._categorical_fields,
            self._continuous_filter_fields,
            self._tooltip_fields,
            self._fillna,
            self._color_by,
        )
        self._cluster_plot = None
        """Index of the current plot being visualized."""

    @property
    def clusters(self):
        if self._cluster_plot is None:
            logging.warning("No clusters have been computed yet.")
            return None
        else:
            return self._cluster_plot._corpora

    def _cluster_button(self) -> widgets.Button:
        """Create a button for clustering the data."""

        # TODO: add selectors for clustering parameters

        def cluster(button):
            # TODO: add clustering parameters

            if self._cluster_plot is None:
                # Initialize clustered plot
                clusters = list(self._plot._corpora[0].cluster())

                if self._keyword_extractor:
                    for c in clusters:
                        c.top_words = self._keyword_extractor.top_words(c)
                self._cluster_plot = PlotWidgets(
                    clusters,
                    self._categorical_fields,
                    self._continuous_filter_fields,
                    self._tooltip_fields,
                    self._fillna,
                    self._color_by,
                )

            widgets = self._cluster_plot._widgets + [self._return_button()]
            clear_output(wait=True)
            display(*widgets)

        button = widgets.Button(
            description="Cluster",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Cluster the data",
            # icon="check",  # (FontAwesome names without the `fa-` prefix)
        )
        button.on_click(cluster)

        return button

    def _return_button(self) -> widgets.Button:
        def _return(button):
            clear_output(wait=True)
            widgets = self._plot._widgets + [self._cluster_button()]
            clear_output(wait=True)
            display(*widgets)

        button = widgets.Button(
            description="Return",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Return to initial view",
        )
        button.on_click(_return)

        return button

    def visualize(self):
        """Display the visualization."""
        widgets = self._plot._widgets + [self._cluster_button()]
        display(*widgets)


class PlotWidgets:
    """A class for holding the widgets for a plot."""

    def __init__(
        self,
        corpora: list[Corpus],
        categorical_fields: list[str],
        continuous_filter_fields: list[str],
        tooltip_fields: list[str],
        fillna: dict[str, str],
        color_by: str,
    ):
        """Create a PlotWidgets object to create the widgets for a JScatterVisualizer.

        Args:
            corpus (Corpus): The corpus to visualize.
            categorical_fields (list[str], optional): The categorical fields to filter on.
            continuous_filter_fields (list[str], optional): The continuous fields to filter on.
            tooltip_fields (list[str], optional): The fields to show in the tooltip.
            fillna (dict[str, str], optional): The values to fill NaN values with.
            color_by (str, optional): The field to color the scatter plot by.
        """

        self._indices: dict[str, pd.RangeIndex] = {}
        """Keeps track of filtered indices per filter field."""

        self._corpora: list[Corpus] = corpora
        self._fillna = fillna
        self._tooltip_fields = tooltip_fields
        self._color_by = color_by

        self._categorical_fields = categorical_fields
        self._continuous_fields = continuous_filter_fields

        self._init_scatter()
        self._init_widgets()

    def __len__(self):
        return len(self._corpora)

    def _init_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame from the corpora."""

        self._df = (
            pd.concat(
                c.to_dataframe().assign(label=c.label).assign(outlier=c.is_outliers())
                for c in self._corpora
            )
            .reset_index()
            .fillna(self._fillna)
            .convert_dtypes()
        )
        return self._df

    def _init_scatter(self) -> jscatter.Scatter:
        """Create the scatter plot."""

        self._scatter = (
            jscatter.Scatter(data=self._init_dataframe(), x="x", y="y")
            .color(by=self._color_by)
            .axes(False)
            .tooltip(True, properties=self._tooltip_fields)
        )
        return self._scatter

    def _init_widgets(self) -> list[widgets.Widget]:
        """Create the widgets for filtering the scatter plot."""

        self._widgets = [self._scatter.show()]
        self._widgets.extend(
            [self._category_field_filter(field) for field in self._categorical_fields]
        )
        self._widgets.extend(
            [self._continuous_field_filter(field) for field in self._continuous_fields]
        )

        return self._widgets

    def _category_field_filter(self, field: str) -> widgets.VBox:
        """Create a selection widget for filtering on a categorical field.

        Args:
            field (str): The field to filter on.

        Returns:
            widgets.VBox: A widget containing the selection widget and the output widget
        """
        # FIXME: this not work for filtering by "top words"

        if field not in self._df.columns:
            logging.warning(f"Categorical field '{field}' not found, ignoring")
            return

        options = self._df[field].unique().tolist()

        if len(options) > 1:
            selector = widgets.SelectMultiple(
                options=options,
                value=options,  # TODO: filter out outliers
                description=field,
                layout={"width": "max-content"},
                rows=len(options),
            )

            selector_output = widgets.Output()

            def handle_change(change):
                self._filter(field, self._df.query(f"{field} in @change.new").index)

            selector.observe(handle_change, names="value")

            return widgets.VBox([selector, selector_output])
        else:
            logging.debug(f"Skipping field {field} with only {len(options)} option(s)")

    def _continuous_field_filter(self, field: str = "year") -> widgets.VBox:
        """Create a selection widget for filtering on a continuous field.

        Args:
            field (str): The field to filter on.
        Returns:
            widgets.VBox: A widget containing a RangeSlider widget and the output widget
        """

        min_year = self._df[field].min()
        max_year = self._df[field].max()

        selection = widgets.SelectionRangeSlider(
            options=[str(i) for i in range(min_year, max_year + 1)],
            index=(0, max_year - min_year),
            description=field,
            continuous_update=True,
        )

        selection_output = widgets.Output()

        def handle_slider_change(change):
            start = int(change.new[0])  # noqa: F841
            end = int(change.new[1])  # noqa: F841

            self._filter(field, self._df.query("year > @start & year < @end").index)

        selection.observe(handle_slider_change, names="value")

        return widgets.VBox([selection, selection_output])

    def _filter(self, field, index):
        """Filter the scatter plot based on the given field and index.

        Intersect the indices per field to get the final index to filter in the plot.

        Args:
            field (str): The field to filter on.
            index (pd.RangeIndex): The index listing the rows to keep for this field
        """
        self._indices[field] = index

        index = self._df.index
        for _index in self._indices.values():
            index = index.intersection(_index)

        self._scatter.filter(index)
