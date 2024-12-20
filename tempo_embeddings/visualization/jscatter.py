import csv
import logging
from typing import Any, Iterable, Optional

import jscatter
import pandas as pd
from IPython.display import display
from ipywidgets import widgets

from ..settings import OUTLIERS_LABEL, STOPWORDS
from ..text.corpus import Corpus
from ..text.keyword_extractor import KeywordExtractor


class JScatterContainer:
    """A container with tabs for JScatterVisualizer objects."""

    def __init__(self, corpora: list[Corpus], **kwargs):
        """Create a JScatterContainer object to visualize a list of corpora.

        Args:
            corpora (list[Corpus]): The corpora to visualize initially.
        KwArgs:
            Arguments to pass to the visualizer, overriding the current values.
        """
        self._tab = widgets.Tab()
        self._visualizer = JScatterVisualizer(corpora, container=self, **kwargs)
        """The root visualizer."""

        self.add_tab(self._visualizer)

    def add_tab(self, visualizer: "JScatterVisualizer", *, title: Optional[str] = None):
        if title is None:
            title = (
                f"Clusters {len(self._tab.children)}"
                if self._tab.children
                else "Full Corpus"
            )

        self._tab.children = (list(self._tab.children) or []) + [
            widgets.VBox(visualizer.get_widgets())
        ]

        self._tab.set_title(-1, title)
        self._tab.selected_index = len(self._tab.children) - 1

    def visualize(self):
        display(self._tab)


class JScatterVisualizer:
    """A class for creating interactive scatter plots with Jupyter widgets."""

    _DEFAULT_CONTINUOUS_FIELDS: set[str] = {"year"}
    _EXCLUDE_FILTER_FIELDS: set[str] = {
        "date",
        "day",
        "month",
        "origin_id",
        "page",
        "sentence_index",
        "sequence",
        "url",
        "year",
    }
    _EXCLUDE_TOOLTIP_FIELDS: set[str] = {"date"}

    _REQUIRED_FIELDS: dict[str, Any] = {"x": pd.Float64Dtype(), "y": pd.Float64Dtype()}
    """Required fields and dtype."""

    def __init__(
        self,
        corpora: list[Corpus],
        *,
        container: Optional[JScatterContainer] = None,
        categorical_fields: Optional[list[str]] = None,
        continuous_fields: list[str] = _DEFAULT_CONTINUOUS_FIELDS,
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

        self._container = container

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

        self._continuous_fields = continuous_fields
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

        self._plot_widgets = self.PlotWidgets(
            df=self._df, color_by=self._color_by, tooltip_fields=self._tooltip_fields
        )

    def _validate_corpora(self, corpora):
        for column in self._REQUIRED_FIELDS:
            if not all(column in c.to_dataframe().columns for c in corpora):
                raise ValueError(f"Missing required field '{column}' in corpora.")

    def _valid_tooltip_fields(self, tooltip_fields: set[str]) -> set[str]:
        return (
            set(tooltip_fields)
            .intersection(self._df.columns)
            .difference(self._EXCLUDE_TOOLTIP_FIELDS)
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
        if "year" not in self._df.columns:
            self._df["year"] = self._df["date"].dt.year
        else:
            self._df.fillna({"year": self._df["date"].dt.year}, inplace=True)

        self._df["date"] = self._df["date"].apply(pd.to_datetime)

        # Validate required fields
        for field, dtype in self._REQUIRED_FIELDS.items():
            if field not in self._df.columns:
                raise ValueError(f"Required field '{field}' not found.")
            if self._df[field].dtype != dtype:
                raise ValueError(
                    f"Field '{field}' has incorrect dtype: {self._df[field].dtype}"
                )
            if self._df[field].isna().any():
                raise ValueError(f"Field '{field}' contains NaN values.")

    def get_widgets(self) -> list[widgets.Widget]:
        """Create all widgets"""

        _widgets: list[widgets.Widget] = self._plot_widgets.get_widgets(
            continuous_fields=self._continuous_fields,
            categorical_fields=self._categorical_fields,
            keyword_extractor=self._keyword_extractor,
            umap_model=self._umap,
        )

        if self._container is None:
            logging.warning("No container set, skipping cluster button.")
        else:
            # insert Cluster button below filter widgets
            _widgets.insert(3, self._cluster_button())

        return _widgets

    def with_corpora(self, corpora: list[Corpus], **kwargs) -> "JScatterVisualizer":
        """Create a new JScatterVisualizer with the given corpora.

        Args:
            corpora (list[Corpus]): The corpora to visualize.
        KwArgs:
            Arguments to pass to the constructor, overriding the current values.

        Returns:
            JScatterVisualizer: A new JScatterVisualizer object.
        """
        visualizer_args = {
            "categorical_fields": self._categorical_fields,
            "continuous_fields": self._continuous_fields,
            "tooltip_fields": self._tooltip_fields,
            "color_by": [self._color_by],
            "keyword_extractor": self._keyword_extractor,
        } | kwargs
        return JScatterVisualizer(corpora, container=self._container, **visualizer_args)

    def _cluster_button(self) -> widgets.Button:
        """Create a button for clustering the data.

        This button triggers the creation of a new set of corpora (the clusters) and adds a new visualizer to the JScatterContainer instance.

        Dev note: that is why this method is part of the JScatterVisualizer class, not the PlotWidgets class.
        """

        def cluster(button):  # pragma: no cover
            button.disabled = True
            button.description = "Clustering..."
            # TODO: add clustering parameters

            clusters = list(
                Corpus.from_dataframe(
                    self._df.loc[self._plot_widgets.selected()], umap_model=self._umap
                ).cluster()
            )

            for c in clusters:
                c.top_words = self._keyword_extractor.top_words(
                    c, use_2d_embeddings=True
                )

            self._container.add_tab(self.with_corpora(clusters, tooltip_fields=None))

            button.disabled = False
            button.description = "Cluster"

        button = widgets.Button(
            description="Cluster",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Cluster the data, create new tab",
            # icon="check",  # (FontAwesome names without the `fa-` prefix)
        )
        button.on_click(cluster)

        return button

    class PlotWidgets:
        """A class for generating the widgets for a plot."""

        __SHOW_ALL: str = "<SHOW ALL>"

        def __init__(
            self, *, df: pd.DataFrame, color_by: str, tooltip_fields: set[str]
        ):
            self._df = df
            self._color_by = color_by
            self._tooltip_fields = tooltip_fields

            self._scatter_plot: jscatter.JScatter = self._scatter()

            self._indices = dict()
            """The indices of the filtered rows per field."""

        def _scatter(self) -> jscatter.Scatter:
            """Create the scatter plot."""

            return (
                jscatter.Scatter(data=self._df, x="x", y="y")
                .color(by=self._color_by)
                .axes(False)
                .tooltip(True, properties=self._tooltip_fields)
                .legend(True)
            )

        def _export_button(self) -> widgets.HBox:
            overwrite = widgets.Checkbox(
                description="Overwrite if file exists", value=False
            )
            csv_file = widgets.Text(
                description="Filename", value="export.csv", disabled=False
            )

            def export(change):
                try:
                    selected: pd.DataFrame = self._df.iloc[self.selected()]
                    selected.to_csv(
                        csv_file.value,
                        columns=[
                            c for c in selected.columns if selected[c].notna().any()
                        ],
                        index=False,
                        quoting=csv.QUOTE_ALL,
                        mode="w" if overwrite.value else "x",
                    )
                except FileExistsError as e:
                    logging.error(e)
                else:
                    overwrite.value = False

            button = widgets.Button(
                description="Export", tooltip="Export selected data points"
            )
            button.on_click(export)

            return widgets.HBox((button, csv_file, overwrite))

        def _color_by_dropdown(self):
            current = self._scatter_plot.color()["by"]
            columns = [
                c
                for c in self._df.columns
                if not self._df[c].hasnans and 1 < self._df[c].unique().size <= 50
            ]
            if current not in columns:
                columns.append(current)
            # TODO: update columns by selection

            def handle_change(change):
                self._scatter_plot.color(by=change["new"], map="auto")

            color_by_dropdown = widgets.Dropdown(
                options=columns, value=current, description="Color by:", disabled=False
            )

            color_by_dropdown.observe(handle_change, names="value")

            return color_by_dropdown

        def _select_tooltips(self):
            def handle_change(change):
                self._scatter_plot.tooltip(enable=True, properties=change["new"])

            current: list[str] = self._scatter_plot.tooltip()["properties"]
            options = sorted(self._df.columns)

            try:
                # TODO: make generic blacklist for columns
                options.remove("ID_DB")
            except ValueError as e:
                logging.debug(e)

            select_tooltips = widgets.SelectMultiple(
                options=options,
                value=current,
                # rows=10,
                description="Tooltip fields",
                disabled=False,
            )
            select_tooltips.observe(handle_change, names="value")

            return select_tooltips

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
                raise ValueError(f"'{field}' does not exist in the data.")

            values = self._df[field].value_counts()
            options = values.where(values > 1).index.tolist()

            if field in self._df.columns and 1 < len(options):
                selector = widgets.SelectMultiple(
                    options=[self.__SHOW_ALL] + options,
                    value=[self.__SHOW_ALL],  # TODO: filter out outliers
                    description=field,
                    # layout=widgets.Layout(width=50),
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

        def _search_filter(self) -> widgets.HBox:
            """Create a search widget for filtering on a field.

            Returns:
                widgets.HBox: A widget containing a search box
            """

            search = widgets.Text(
                description="Search:",
                placeholder="Enter search term",
                continuous_update=False,
            )
            field_selector = widgets.Dropdown(
                options=sorted(
                    [c for c in self._df.columns if self._df[c].dtype == "string"]
                ),
                value="text",
                description="In field",
                layout={"width": "max-content"},
            )
            _widgets = [search, field_selector]

            def handle_search_change(change):
                for w in _widgets:
                    w.disabled = True

                search_term = search.value.strip()
                if search_term:
                    filtered = self._df.loc[
                        self._df[field_selector.value].str.contains(search_term)
                    ]
                else:
                    filtered = self._df

                self._filter(field_selector, filtered.index)

                for w in _widgets:
                    w.disabled = False

            search.observe(handle_search_change, names="value")
            field_selector.observe(handle_search_change, names="value")

            return widgets.HBox(_widgets)

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

            self._scatter_plot.filter(index)

        def selected(self) -> list[int]:
            """Return the indices of currently selected/filtered/all rows.

            Returns:
                The indices of the selected/filtered/all rows.
            """
            if len(self._scatter_plot.selection()) > 0:
                index = self._scatter_plot.selection()
            else:
                try:
                    # this should be identical with the intersection of all _indices values
                    filter_indices = self._scatter_plot.filter()
                except AttributeError:
                    # filter() does not exist if not filter has been set yet
                    index = self._df.index
                else:
                    index = (
                        filter_indices
                        if filter_indices is not None and filter_indices.size > 0
                        else self._df.index
                    )
            return index

        def _plot_by_field_button(self) -> widgets.Button:
            field = "year"

            window_size_slider = widgets.BoundedIntText(
                value=5,
                min=1,
                step=1,
                description="Rolling Window over Years:",
                layout={"width": "max-content"},
            )

            options = [c for c in self._df.columns if c not in {field, "text"}]
            value = next(
                (c for c in ["label", "cluster", "corpus"] if c in options), options[0]
            )

            # TODO: update option to match selection
            groups_field_selector = widgets.Dropdown(
                description="Field to plot",
                options=options,
                value=value,
                layout={"width": "max-content"},
            )
            button = widgets.Button(
                description="Plot by Field",
                tooltip="Plot (selected) frequencies over years by selected field",
            )
            box_widgets = (button, window_size_slider, groups_field_selector)

            corpus_per_year = self._df[field].value_counts()

            def _plot_by_field(b):
                _selection = self._df.loc[self.selected()]
                groups_field = groups_field_selector.value

                n_values = _selection[groups_field].unique().size
                if n_values > 20:
                    logging.warning(
                        f"Too many groups to plot for field '{groups_field}' ({n_values})."
                    )
                elif groups_field in _selection.columns:
                    for widget in box_widgets:
                        widget.disabled = True
                    button.description = "Plotting..."

                    for label, group in _selection.groupby(groups_field):
                        window = window_size_slider.value
                        if label != OUTLIERS_LABEL:
                            _series = (
                                (group[field].value_counts() / corpus_per_year)
                                .sort_index()
                                .rolling(window)
                                .mean()
                            )
                            _series.name = label
                            ax = _series.plot(kind="line", legend=label)
                            ax.set_title(
                                f"Relative Frequency by '{groups_field}' (Rolling Window over {window} {field}s)"
                            )
                            ax.set_xlabel(field)
                            ax.set_ylabel("Relative Frequency")

                    for widget in box_widgets:
                        widget.disabled = False
                    button.description = "Plot by Field"
                else:
                    # TODO: this should never happen if the dropdown is updated
                    raise ValueError(f"Field '{groups_field}' not found in selection.")

            button.on_click(_plot_by_field)

            return widgets.HBox(box_widgets)

        def _top_words_button(
            self, keyword_extractor: KeywordExtractor, umap_model
        ) -> widgets.Button:
            text = widgets.Text(
                description="Top Words:",
                disabled=True,
                placeholder="Top words for current selection will appear here.",
                layout=widgets.Layout(width="100%", height="100%"),
            )

            def _show_top_words(b):  # pragma: no cover
                button.disabled = True
                button.description = "Calculating..."
                text.value = ""

                corpus = Corpus.from_dataframe(
                    self._df.loc[self.selected()], umap_model=umap_model
                )
                top_words = keyword_extractor.top_words(corpus, use_2d_embeddings=True)
                text.value = "; ".join(top_words)
                button.disabled = False
                button.description = "Top words:"

            button = widgets.Button(
                description="Top words",
                disabled=False,
                button_style="",  # 'success', 'info', 'warning', 'danger' or ''
                tooltip="Compute top words for current selection",
            )
            button.on_click(_show_top_words)

            return widgets.HBox((button, text))

        def get_widgets(
            self,
            *,
            continuous_fields: Iterable[str],
            categorical_fields: Iterable[str],
            keyword_extractor: KeywordExtractor,
            umap_model,
        ) -> list[widgets.Widget]:
            """Create all widgets

            Args:
                continuous_fields (Iterable[str]): The continuous fields to filter on.
                categorical_fields (Iterable[str]): The categorical fields to filter on.
                keyword_extractor (KeywordExtractor): The keyword extractor to use for generating top words.
                umap_model: The UMAP model to use for generating top words.

            Returns:
                list[widgets.Widget]: The widgets to display.
            """
            # TODO: move the arguments to the class level

            continuous_filters: list[widgets.Widget] = [
                self._continuous_field_filter(field) for field in continuous_fields
            ]
            category_filters: list[Optional[widgets.Widget]] = [
                self._category_field_filter(field)
                for field in categorical_fields
                if field not in JScatterVisualizer._EXCLUDE_FILTER_FIELDS
            ]

            return [self._scatter_plot.show()] + [
                widgets.HBox(continuous_filters),
                widgets.HBox(
                    [widget for widget in category_filters if widget is not None]
                ),
                self._search_filter(),
                self._color_by_dropdown(),
                self._select_tooltips(),
                self._export_button(),
                self._top_words_button(keyword_extractor, umap_model),
                self._plot_by_field_button(),
            ]
