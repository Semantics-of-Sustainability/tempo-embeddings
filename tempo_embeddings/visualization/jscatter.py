import csv
import logging
from typing import Any, Optional

import jscatter
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import widgets

from ..settings import STOPWORDS
from ..text.corpus import Corpus
from ..text.keyword_extractor import KeywordExtractor
from .util import DownloadButton


class JScatterVisualizer:
    """A class for creating interactive scatter plots with Jupyter widgets."""

    __REQUIRED_FIELDS: dict[str, Any] = {"x": pd.Float64Dtype(), "y": pd.Float64Dtype()}
    """Required fields and dtype."""

    __DEFAULT_CONTINUOUS_FIELDS: set[str] = {"year"}
    __EXCLUDE_FILTER_FIELDS: set[str] = {"month", "day", "year"}
    __EXCLUDE_TOOLTIP_FIELDS: set[str] = {"date"}

    def __init__(
        self,
        corpora: list[Corpus],
        *,
        categorical_fields: Optional[list[str]] = None,
        continuous_filter_fields: list[str] = __DEFAULT_CONTINUOUS_FIELDS,
        tooltip_fields: list[str] = None,
        color_by: list[str] = ["cluster", "label"],
        keyword_extractor: Optional[KeywordExtractor] = None,
    ):
        """Create a JScatterVisualizer object to visualize a corpus.

        Args:
            corpus (Corpus): The corpus to visualize.
            categorical_fields (list[str], optional): The categorical fields to filter on. Defaults to all metadata fields minus the continus fields.
            continuous_filter_fields (list[str], optional): The continuous fields to filter on. Defaults to DEFAULT_CONTINUOUS_FIELDS.
            tooltip_fields (list[str], optional): The fields to show in the tooltip. Defaults to all metadata fields.
            color_by (str, optional): The field to color the scatter plot by.
            keyword_extractor (KeywordExtractor, optional): The keyword extractor to use.

        """
        self._validate_corpora(corpora)
        self._corpora = corpora

        self._umap = corpora[0].umap
        """Common UMAP model; assuming all corpora have the same model."""

        merged_corpus = sum(corpora, Corpus())
        self._keyword_extractor = keyword_extractor or KeywordExtractor(
            merged_corpus, exclude_words=STOPWORDS
        )

        self._init_df()

        self._tooltip_fields: set[str] = self._valid_tooltip_fields(
            tooltip_fields or merged_corpus.metadata_fields()
        )

        self._continuous_fields = continuous_filter_fields
        self._categorical_fields = categorical_fields or (
            merged_corpus.metadata_fields() | {"label"} - set(self._continuous_fields)
        )

        try:
            self._color_by = next(
                column for column in color_by if column in self._df.columns
            )
        except StopIteration as e:
            raise ValueError(
                f"None of the color_by fields found in corpus: {color_by}"
            ) from e

        self._plot_widgets = self.PlotWidgets(self)
        self._scatter = self._plot_widgets._scatter()

    def _validate_corpora(self, corpora):
        for column in self.__REQUIRED_FIELDS:
            if not all(column in c.to_dataframe().columns for c in corpora):
                raise ValueError(f"Missing required field '{column}' in corpora.")

    def _valid_tooltip_fields(self, tooltip_fields: set[str]) -> set[str]:
        return (
            set(tooltip_fields)
            .intersection(self._df.columns)
            .difference(self.__EXCLUDE_TOOLTIP_FIELDS)
            .difference(
                (
                    column
                    for column, dtype in self._df.dtypes.items()
                    if dtype == "object"
                )
            )
        )

    def _init_df(self):
        self._df = (
            pd.concat(c.to_dataframe().assign(label=c.label) for c in self._corpora)
            .convert_dtypes()
            .reset_index()
        )

        # Fill missing 'year' values from 'date' field:
        self._df.fillna({"year": self._df["date"].dt.year}, inplace=True)

        self._df["date"] = self._df["date"].apply(pd.to_datetime)

        # Validate required fields
        for field, dtype in self.__REQUIRED_FIELDS.items():
            if field not in self._df.columns:
                raise ValueError(f"Required field '{field}' not found.")
            if self._df[field].dtype != dtype:
                raise ValueError(
                    f"Field '{field}' has incorrect dtype: {self._df[field].dtype}"
                )
            if self._df[field].isna().any():
                raise ValueError(f"Field '{field}' contains NaN values.")

    def _selected(self) -> list[int]:
        """Return the indices of currently selected/filtered/all rows.

        Returns:
            The indices of the selected/filtered/all rows.
        """
        if self._scatter.selection().size > 0:
            index = self._scatter.selection()
        else:
            try:
                # this should be identical with the intersection of all _indices values
                filter_indices = self._scatter.filter()
            except AttributeError:
                # filter() raises error if it has not been set yet
                logging.debug("No filter set.")
                index = self._df.index
            else:
                index = filter_indices if filter_indices.size > 0 else self._df.index
        return index

    def with_corpora(self, corpora: list[Corpus], **kwargs) -> "JScatterVisualizer":
        """Create a new JScatterVisualizer with the given corpora.

        Args:
            corpora (list[Corpus]): The corpora to visualize.
        KwArgs:
            Arguments to pass to the constructor, overriding the current values.

        Returns:
            JScatterVisualizer: A new JScatterVisualizer object.
        """
        args = {
            "categorical_fields": self._categorical_fields,
            "continuous_filter_fields": self._continuous_fields,
            "tooltip_fields": self._tooltip_fields,
            "color_by": [self._color_by],
            "keyword_extractor": self._keyword_extractor,
        } | kwargs
        return JScatterVisualizer(corpora, **args)

    def visualize(self) -> None:
        """Display the initial visualization."""
        continuous_filters: list[widgets.Widget] = [
            self._plot_widgets._continuous_field_filter(field)
            for field in self._continuous_fields
        ]
        category_filters: list[widgets.Widget] = [
            self._plot_widgets._category_field_filter(field)
            for field in self._categorical_fields
            if field not in self.__EXCLUDE_FILTER_FIELDS
        ]

        _widgets: list[widgets.Widget] = [self._scatter.show()] + [
            widgets.HBox(continuous_filters),
            widgets.HBox([widget for widget in category_filters if widget is not None]),
            self._plot_widgets._cluster_button(),
            self._plot_widgets._export_button(),
            # self._top_words_button(),
        ]
        display(*_widgets)

    class PlotWidgets:
        """A class for generating the widgets for a plot."""

        __SHOW_ALL: str = "<SHOW ALL>"

        def __init__(self, visualizer: "JScatterVisualizer"):
            self._visualizer = visualizer
            self._df = self._visualizer._df

            self._indices = dict()
            """The indices of the filtered rows per field."""

        def _scatter(self) -> jscatter.Scatter:
            """Create the scatter plot."""

            return (
                jscatter.Scatter(data=self._visualizer._df, x="x", y="y")
                .color(by=self._visualizer._color_by)
                .axes(False)
                .tooltip(True, properties=self._visualizer._tooltip_fields)
                .legend(True)
            )

        def _cluster_button(self) -> widgets.Button:
            """Create a button for clustering the data."""

            # TODO: add selectors for clustering parameters

            def cluster(button):  # pragma: no cover
                # TODO: add clustering parameters

                clusters = list(
                    Corpus.from_dataframe(
                        self._df.iloc[self._visualizer._selected()],
                        umap_model=self._visualizer._umap,
                    ).cluster()
                )

                if self._visualizer._keyword_extractor:
                    for c in clusters:
                        c.top_words = self._visualizer._keyword_extractor.top_words(
                            c, use_2d_embeddings=True
                        )

                # TODO: visualize in new tab widget
                self._visualizer.with_corpora(clusters, tooltip_fields=None).visualize()

                # display(*widgets, clear=True)

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

                display(*widgets, clear=True)

            button = widgets.Button(
                description="Return",
                disabled=False,
                button_style="",  # 'success', 'info', 'warning', 'danger' or ''
                tooltip="Return to initial view",
            )
            button.on_click(_return)

            return button

        def _export_button(self) -> DownloadButton:
            def selected_rows():
                return self._df.iloc[self._visualizer._selected()].to_csv(
                    index=False, quoting=csv.QUOTE_ALL
                )

            return DownloadButton(
                filename="scatter_plot.csv",
                contents=selected_rows,
                description="Export",
            )

        def _top_words_button(self) -> widgets.Button:
            def _show_top_words(b):
                # TODO: create a link between self._df and the corpora
                # TODO: keep/unify text column names
                # Corpus.from_csv_stream(self._df.iloc[self._scatter.selection()].to_csv())
                corpus = Corpus.from_dataframe(self._df[self._selected()])
                top_words = self._keyword_extractor.top_words(corpus)
                print(top_words)

            button = widgets.Button(
                description="Top words",
                disabled=False,
                button_style="",  # 'success', 'info', 'warning', 'danger' or ''
                tooltip="Show top words",
            )
            button.on_click(_show_top_words)
            return button

        def _category_field_filter(
            self, field: str
        ) -> Optional[widgets.SelectMultiple]:
            """Create a selection widget for filtering on a categorical field.

            Args:
                field (str): The field to filter on.

            Returns:
                widgets.SelectMultiple: A widget containing the selection widget or None if the field is not suitable for filtering.
            """
            # FIXME: this does not work for filtering by "top words"

            if field not in self._df.columns:
                raise ValueError(f"'{field}' does not exist.")
            options = self._df[field].dropna().unique().tolist()

            if field in self._df.columns and 1 < len(options) <= 50:
                selector = widgets.SelectMultiple(
                    options=[self.__SHOW_ALL] + options,
                    value=[self.__SHOW_ALL],  # TODO: filter out outliers
                    description=field,
                    layout={"width": "max-content"},
                    rows=min(len(options) + 1, 10),
                )

                def handle_change(change):
                    if self.__SHOW_ALL in change.new:
                        filtered = self._df
                    else:
                        filtered = self._df.loc[self._df[field].isin(change.new)]

                    self._filter(field, filtered.index)

                selector.observe(handle_change, names="value")

                widget = selector
            else:
                logging.info(
                    f"Skipping field {field} with {len(options)} option(s) for filtering."
                )
                widget = None
            return widget

        def _continuous_field_filter(self, field: str) -> widgets.SelectionRangeSlider:
            """Create a selection widget for filtering on a continuous field.

            Args:
                field (str): The field to filter on.
            Returns:
                widgets.SelectionRangeSlider: A widget containing a RangeSlider
            """
            if field not in self._df.columns:
                raise ValueError(f"Field '{field}' not found.")

            min_value = self._df[field].min()
            max_value = self._df[field].max()

            selection = widgets.SelectionRangeSlider(
                options=[str(i) for i in range(min_value, max_value + 1)],
                index=(0, max_value - min_value),
                description=field,
                continuous_update=True,
            )

            def handle_slider_change(change):
                filtered = self._df.loc[
                    (
                        (self._df[field] >= int(change.new[0]))
                        & (self._df[field] < int(change.new[1]))
                    )
                    | self._df[field].isna()
                ]
                self._filter(field, filtered.index)

            selection.observe(handle_slider_change, names="value")

            return selection

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

            self._visualizer._scatter.filter(index)
