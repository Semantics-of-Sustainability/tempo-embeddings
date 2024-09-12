import logging
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from umap.umap_ import UMAP

from tempo_embeddings.settings import STRICT
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestCorpus:
    def test_add(self):
        assert Corpus([Passage("test1")]) + Corpus([Passage("test2")]) == Corpus(
            [Passage("test1"), Passage("test2")]
        )

    def test_extend(self):
        corpus = Corpus([Passage("test1")])
        assert corpus.extend([Passage("test2")]) == range(1, 2)
        assert corpus.passages == [Passage("test1"), Passage("test2")]

    @pytest.mark.parametrize(
        "passages, key, expected",
        [
            ([], "key", []),
            (
                [Passage("test text", metadata={"key": 1})],
                "key",
                [(1, [Passage("test text", metadata={"key": 1})])],
            ),
            (
                [Passage("test text", metadata={"key": 1})],
                "unknown key",
                [(None, [Passage("test text", metadata={"key": 1})])],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"key": 1}),
                ],
                "key",
                [
                    (
                        1,
                        [
                            Passage("test text 1", metadata={"key": 1}),
                            Passage("test text 2", metadata={"key": 1}),
                        ],
                    )
                ],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"key": 2}),
                ],
                "key",
                [
                    (1, [Passage("test text 1", metadata={"key": 1})]),
                    (2, [Passage("test text 2", metadata={"key": 2})]),
                ],
            ),
        ],
    )
    def test_groupby(self, passages, key, expected):
        groups = Corpus(passages).groupby(key)

        for (group, elements), (expected_group, expected_elements) in zip(
            groups, expected, **STRICT
        ):
            assert group == expected_group
            assert list(elements) == expected_elements

    @pytest.mark.parametrize(
        "n_passages,max_clusters,min_cluster_size",
        [(10, 10, 2), (50, 3, 2), (50, 10, 2), (50, 5, 5)],
    )
    def test_cluster(self, caplog, n_passages, max_clusters, min_cluster_size):
        corpus = Corpus(
            [
                Passage(f"test {str(i)}", embedding=np.random.rand(768))
                for i in range(n_passages)
            ]
        )

        with caplog.at_level(logging.WARNING):
            clusters = corpus.cluster(
                max_clusters=max_clusters, min_cluster_size=min_cluster_size
            )

        # FIXME: clustering is not deterministic
        assert (
            len(clusters) <= max_clusters
            or ("root", logging.WARNING, "Could not reduce number of clusters.")
            in caplog.record_tuples
        )

        for cluster in clusters:
            assert cluster._parent_corpus is corpus
            assert len(cluster) >= min_cluster_size or cluster._label == -1

    @pytest.mark.parametrize(
        "corpus,metadata_fields,expected",
        [
            (Corpus(), None, []),
            (Corpus([Passage("test")]), None, [{"text": "test", "corpus": "None"}]),
            (
                Corpus([Passage("test")], label="test label"),
                None,
                [{"text": "test", "corpus": "test label"}],
            ),
        ],
    )
    def test_hover_datas(self, corpus, metadata_fields, expected):
        assert corpus.hover_datas(metadata_fields=metadata_fields) == expected

    @pytest.mark.parametrize(
        "corpus,key,expected",
        [
            (Corpus(), "test key", []),
            (Corpus(passages=[Passage("text", metadata={"key": 1})]), "key", [1]),
            # FIXME: redundant test cases since passages are not filtered by highlighting
            (
                Corpus(
                    [
                        Passage(
                            "text",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        )
                    ],
                ),
                "key",
                [1],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage(
                            "text 2",
                            metadata={"key": 2, "other": 2},
                            highlighting=Highlighting(5, 6),
                        ),
                    ],
                ),
                "key",
                [1, 2],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage(
                            "text 2",
                            metadata={"other": 2},
                            highlighting=Highlighting(5, 6),
                        ),
                    ],
                ),
                "key",
                [1, None],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage("text 2", metadata={"key": 2, "other": 2}),
                    ],
                ),
                "key",
                [1, 2],
            ),
        ],
    )
    def test_get_token_metadatas(self, corpus: Corpus, key, expected):
        assert list(corpus.get_metadatas(key)) == expected

    def test_load_save(self, tmp_path):
        # TODO: test/handle model (de)serialization
        filepath = tmp_path / "corpus"
        corpus = Corpus(
            [
                Passage("text 1", metadata={"key": 1, "other": 3}),
                Passage("text 2", metadata={"key": 2}),
            ],
            [Highlighting(0, 4)],
        )
        corpus.save(filepath)

        assert filepath.is_file()
        assert Corpus.load(filepath) == corpus

    @pytest.mark.parametrize(
        "corpus,expected",
        [
            (Corpus([Passage("test")]), np.array([None])),
            (Corpus([Passage("test", embedding=np.ones(10))]), np.ones((1, 10))),
            (
                Corpus(
                    [
                        Passage("test 1", embedding=np.ones(10)),
                        Passage("test 2", embedding=np.ones(10)),
                    ]
                ),
                np.ones((2, 10)),
            ),
        ],
    )
    def test_embeddings(self, corpus, expected):
        assert_equal(corpus.embeddings, expected)

    @pytest.mark.parametrize(
        "corpus",
        [
            (Corpus([Passage("test")])),
            (Corpus([Passage("test" + str(i)) for i in range(5)])),
        ],
    )
    def test_embeddings_setter(self, corpus):
        embeddings = [np.random.rand(10) for passage in corpus.passages]
        corpus.embeddings = np.array(embeddings)
        for passage, embedding in zip(corpus.passages, embeddings):
            assert_equal(passage.embedding, embedding)

    @pytest.mark.parametrize(
        "embeddings,expected",
        [
            (np.ones((1, 2)), pd.DataFrame({"x": [1], "y": [1]}, dtype=np.float64)),
            (
                np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64),
                pd.DataFrame({"x": [1, 3, 5], "y": [2, 4, 6]}, dtype=np.float64),
            ),
        ],
    )
    def test_coordinates(self, embeddings, expected):
        corpus = Corpus([Passage("test" + str(i)) for i in range(embeddings.shape[0])])
        corpus.embeddings_2d = embeddings
        pd.testing.assert_frame_equal(corpus.coordinates(), expected)

    @pytest.mark.parametrize(
        "sample_size, centroid_based_sample, expected_exception",
        [
            (None, True, pytest.raises(ValueError)),
            (None, False, None),
            (1, True, None),
            (2, False, None),
            (2, False, None),
        ],
    )
    def test_to_dataframe(self, sample_size, centroid_based_sample, expected_exception):
        embeddings = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        corpus = Corpus([Passage("test" + str(i)) for i in range(embeddings.shape[0])])
        corpus.embeddings = embeddings

        # emulate compressed 2d embeddings
        corpus._embeddings_2d = embeddings.copy()
        corpus._umap = UMAP()  # Set unfitted dummy UMAP to prevent re-computation

        with expected_exception or does_not_raise():
            df = corpus.to_dataframe(sample_size, centroid_based_sample)

        if expected_exception is None:
            if sample_size is None:
                assert df.shape[0] == len(embeddings)
            elif centroid_based_sample:
                assert (
                    df.iloc[0]["x"] == embeddings[1][0]
                    and df.iloc[0]["y"] == embeddings[1][1]
                )
            else:
                assert df.shape[0] == sample_size

    @pytest.mark.parametrize(
        "passages,expected",
        [
            ([], False),
            ([Passage("test")], False),
            ([Passage("test", embedding=np.random.rand(10))], True),
            (
                [
                    Passage("test 1", embedding=np.random.rand(10)),
                    Passage("test 2", embedding=np.random.rand(10)),
                ],
                True,
            ),
        ],
    )
    def test_has_embeddings(self, passages, expected):
        assert Corpus(passages).has_embeddings() == expected

    @pytest.mark.parametrize(
        "corpus, expected_exception",
        [
            (Corpus(), pytest.raises(ValueError)),
            (Corpus([Passage("test text")]), pytest.raises(ValueError)),
            (Corpus([Passage(f"test text {str(i)}") for i in range(10)]), None),
            (Corpus([Passage(f"test text {str(i)}") for i in range(10)]), None),
        ],
    )
    def test_compress_embeddings(self, corpus: Corpus, expected_exception):
        # generate random embedding vectors
        for passage in corpus.passages:
            passage.embedding = np.random.rand(10).tolist()

        with expected_exception or does_not_raise():
            compressed = corpus.compress_embeddings()

        if expected_exception is None:
            assert compressed.shape == (len(corpus), 2)

            np.testing.assert_array_equal(corpus.embeddings_2d, compressed)

            assert corpus.embeddings.shape == (len(corpus), 10)

            if len(corpus) == 1:
                # For single samples, the output should be zeros
                assert_equal(compressed, np.zeros((len(corpus), 2)))
