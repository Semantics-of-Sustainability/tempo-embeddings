import pytest
from ipywidgets.widgets import HBox, VBox

from tempo_embeddings.settings import STRICT
from tempo_embeddings.visualization.jscatter import JScatter


class TestJScatter:
    @pytest.mark.parametrize(
        "categorical_fields, continuous_filter_fields",
        [(["provenance"], ["year"]), (["provenance"], []), ([], ["year"])],
    )
    def test_get_widgets(self, corpus, categorical_fields, continuous_filter_fields):
        widgets = JScatter(
            corpus,
            categorical_fields=categorical_fields,
            continuous_filter_fields=continuous_filter_fields,
        ).get_widgets()

        assert isinstance(
            widgets.pop(0), HBox
        ), "First widget should be an HBox (the Scatter plot)"

        for widget, _ in zip(
            widgets, categorical_fields + continuous_filter_fields, **STRICT
        ):
            assert isinstance(
                widget, VBox
            ), "There should be a VBox widget for each field"
