from bokeh.models.widgets.sliders import RangeSlider

from tempo_embeddings.visualization.bokeh import BokehInteractiveVisualizer


class TestBokehInteractiveVisualizer:
    def test_create_layout(self, corpus):
        visualizer = BokehInteractiveVisualizer(corpus)
        layout = visualizer._create_layout()
        assert layout.children == [visualizer._figure]

    def test_create_layout_years(self, corpus):
        year = 1999
        for passage in corpus.passages:
            passage.metadata["year"] = year

        visualizer = BokehInteractiveVisualizer(corpus)
        layout = visualizer._create_layout()
        assert layout.children[0] == visualizer._figure
        assert isinstance(layout.children[1], RangeSlider)
