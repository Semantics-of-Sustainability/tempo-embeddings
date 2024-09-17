import numpy as np
from bokeh.models.widgets.sliders import RangeSlider

from tempo_embeddings.visualization.bokeh import BokehInteractiveVisualizer


class TestBokehInteractiveVisualizer:
    def test_create_layout(self, corpus):
        corpus.embeddings_2d = np.random.rand(len(corpus), 2)
        visualizer = BokehInteractiveVisualizer(corpus)

        layout = visualizer._create_layout()
        assert layout.children[0] == visualizer._figure
        assert isinstance(layout.children[1], RangeSlider)
