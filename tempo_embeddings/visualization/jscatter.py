import jscatter
import pandas as pd
from ipywidgets import widgets


class JScatter:
    """A class for creating interactive scatter plots with Jupyter widgets."""

    def __init__(
        self,
        *corpora,
        categorical_fields: list[str] = ["newspaper", "label"],
        continuous_filter_fields: list[str] = ["year"],
        tooltip_fields: list[str] = ["year", "text", "label", "top words", "newspaper"],
        fillna: dict[str, str] = {"newspaper": "NRC"},
        color_by: str = "label",
    ):
        self._indices: dict[str, pd.RangeIndex] = {}
        """Keep track of filtered indices per filter field."""

        self._init_dataframe(*corpora, fillna=fillna)
        self._init_scatter(tooltip_fields, color_by)

        self._categorical_fields = categorical_fields
        self._continuous_fields = continuous_filter_fields

    def _init_dataframe(self, *corpora, fillna: dict[str, str]):
        """Create a DataFrame from the given corpora."""
        self._df = (
            pd.concat(
                c.to_dataframe().assign(label=c.label).assign(outlier=c.is_outliers())
                for c in corpora
            )
            .reset_index()
            .fillna(fillna)
            .convert_dtypes()
        )

    def _init_scatter(self, tooltip_fields: list[str], color_by: str):
        """Create the scatter plot."""

        self._scatter = (
            jscatter.Scatter(data=self._df, x="x", y="y")
            .color(by=color_by)
            .axes(False)
            .tooltip(True, properties=tooltip_fields)
        )

        # self._filter("outlier", self._df.index.difference(self._df.query("outlier")))

    def _category_field_filter(self, field: str) -> widgets.VBox:
        """Create a selection widget for filtering on a categorical field.


        Args:
            field (str): The field to filter on.

        Returns:
            widgets.VBox: A widget containing the selection widget and the output widget
        """

        options = self._df[field].unique().tolist()

        selector = widgets.SelectMultiple(
            options=options, value=options, description=field
        )

        selector_output = widgets.Output()

        def handle_change(change):
            self._filter(field, self._df.query(f"{field} in @change.new").index)

        selector.observe(handle_change, names="value")

        return widgets.VBox([selector, selector_output])

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

    def get_widgets(self) -> list[widgets.Widget]:
        return (
            [self._scatter.show()]
            + [
                self._continuous_field_filter(field)
                for field in self._continuous_fields
            ]
            + [self._category_field_filter(field) for field in self._categorical_fields]
        )
