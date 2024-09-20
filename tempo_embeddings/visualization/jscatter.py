from typing import Iterable

import jscatter
import pandas as pd
from ipywidgets import widgets


class JScatter:
    """A class for creating interactive scatter plots with Jupyter widgets."""

    def __init__(
        self,
        *corpora,
        tooltip_fields: list[str] = ["year", "text", "label", "top words", "newspaper"],
        fillna: dict[str, str] = {"newspaper": "NRC"},
        color_by: str = "label",
    ):
        self._indices: dict[str, pd.RangeIndex] = {}
        """Keep track of filtered indices per filter field."""

        self._init_dataframe(*corpora, fillna=fillna)
        self._init_scatter(tooltip_fields, color_by)

    def _init_dataframe(self, *corpora, fillna: dict[str, str]):
        """Create a DataFrame from the given corpora."""
        self._df = (
            pd.concat(c.to_dataframe().assign(label=c.label) for c in corpora)
            .reset_index()
            .fillna(fillna)
            .convert_dtypes()
        )

    def _init_scatter(self, tooltip_fields: list[str], color_by: str):
        """Create the scatter plot."""

        self._scatter = jscatter.Scatter(
            data=self._df,
            x="x",
            y="y",
            color_by=color_by,
            # title=f"Clusters for '{text_widget.value}'",
            legend=True,
        )
        self._scatter.axes(False)
        self._scatter.tooltip(True, properties=tooltip_fields)

        # TODO: grey out Outliers

    def _year_filter(self, *, field: str = "year"):
        """Create a selection widget for filtering on the year field."""

        min_year = self._df[field].min()
        max_year = self._df[field].max()

        year_selection = widgets.SelectionRangeSlider(
            options=[str(i) for i in range(min_year, max_year + 1)],
            index=(0, max_year - min_year),
            description=field,
            continuous_update=True,
        )

        year_selection_output = widgets.Output()

        def handle_slider_change(change):
            start = int(change.new[0])  # noqa: F841
            end = int(change.new[1])  # noqa: F841

            self._filter(field, self._df.query("year > @start & year < @end").index)

        year_selection.observe(handle_slider_change, names="value")

        return widgets.VBox([year_selection, year_selection_output])

    def _news_paper_filter(self, *, field: str = "newspaper"):
        """Create a selection widget for filtering on the newspaper field."""

        options = self._df[field].unique().tolist()

        newspaper_selector = widgets.SelectMultiple(
            options=options, value=options, description=field
        )

        newspaper_selector_output = widgets.Output()

        def handle_newspaper_change(change):
            self._filter(field, self._df.query(f"{field} in @change.new").index)

        newspaper_selector.observe(handle_newspaper_change, names="value")

        return widgets.VBox([newspaper_selector, newspaper_selector_output])

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

    def get_widgets(self) -> Iterable[widgets.Widget]:
        return self._scatter.show(), self._year_filter(), self._news_paper_filter()
