import logging
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from tempo_embeddings.settings import STRICT
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestCorpus:
    def test_add(self, test_passages):
        expected = Corpus(test_passages[:2], None, umap_model=None, vectorizer=None)
        assert Corpus([test_passages[0]]) + Corpus([test_passages[1]]) == expected

    def test_add_umap_fitted(self, corpus):
        corpus._fit_umap()

        with pytest.raises(RuntimeError):
            corpus + Corpus([Passage("test")])

    def test_add_vectorizer_fitted(self, corpus):
        corpus._fit_vectorizer()

        with pytest.raises(RuntimeError):
            corpus + Corpus([Passage("test")])

    @pytest.mark.parametrize(
        "passages, key, default_value, expected",
        [
            ([], "key", None, []),
            (
                [Passage("test text", metadata={"key": 1})],
                "key",
                None,
                [(1, [Passage("test text", metadata={"key": 1})])],
            ),
            (
                [Passage("test text", metadata={"key": 1})],
                "unknown key",
                None,
                [(None, [Passage("test text", metadata={"key": 1})])],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"key": 1}),
                ],
                "key",
                None,
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
                None,
                [
                    (1, [Passage("test text 1", metadata={"key": 1})]),
                    (2, [Passage("test text 2", metadata={"key": 2})]),
                ],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"other key": 1}),
                ],
                "key",
                0,
                [
                    (0, [Passage("test text 2", metadata={"other key": 1})]),
                    (1, [Passage("test text 1", metadata={"key": 1})]),
                ],
            ),
        ],
    )
    def test_groupby(self, passages, key, default_value, expected):
        groups = Corpus(passages).groupby(key, default_value=default_value)

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
            clusters = list(
                corpus.cluster(
                    max_clusters=max_clusters, min_cluster_size=min_cluster_size
                )
            )

        # FIXME: clustering is not deterministic
        assert (
            len(clusters) <= max_clusters
            or ("root", logging.WARNING, "Could not reduce number of clusters.")
            in caplog.record_tuples
        )

        for cluster in clusters:
            assert all(passage in corpus.passages for passage in cluster.passages)
            assert len(cluster) >= min_cluster_size or cluster._label == -1

    @pytest.mark.parametrize(
        "sample_size, centroid_based, exception",
        [(5, False, None), (0, False, None), (10, False, pytest.raises(ValueError))],
    )
    def test_sample(self, corpus, sample_size, centroid_based, exception):
        with exception or does_not_raise():
            sample = corpus.sample(sample_size, centroid_based=centroid_based)

            assert len(sample) == sample_size

            if centroid_based:
                # TODO: test that samples are closest to the centroid
                pass

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
    def test_get_metadatas(self, corpus: Corpus, key, expected):
        assert list(corpus.get_metadatas(key)) == expected

    @pytest.mark.xfail(reason="Not implemented", raises=NotImplementedError)
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

    @pytest.mark.parametrize("normalize", [True, False])
    def test_distances(self, corpus, normalize):
        distances = corpus.distances(normalize=normalize)

        # TODO test embeddings are random, so distances vary
        np.testing.assert_allclose(
            distances, np.zeros(len(corpus)), atol=1.0, verbose=True
        )

    def test_to_dataframe(self, corpus):
        ids = [
            "d4e730f8574317e635c56793c11044a23b25ec6e22bb5c0ec52c99fef80a3957",
            "4d9bf2e19ace7ed314607c76e912e3dd33905f356a9699ef71a02cda6edc68a6",
            "aa71b4bc2bdc330d59a304adf67c3379ecbf7e2e8d62be178ea1afcb8087572a",
            "a2e60054d9450b163a9045e25a0525a2d2e76f7e1fcab5faafbb85368b065f2d",
            "85bde0f0f0eb6fc2b9b115f536a0780a5322d270e57c528901c3d5da3aa4990e",
        ]
        expected = pd.DataFrame(
            [
                {
                    "text": passage.text,
                    "ID_DB": _id,
                    "highlight_start": 1,
                    "highlight_end": 3,
                    "provenance": "test_file",
                    "x": 0.0,
                    "y": 0.0,
                    "distance_to_centroid": 0.0,
                }
                for passage, _id in zip(corpus.passages, ids)
            ]
        )

        pd.testing.assert_frame_equal(corpus.to_dataframe(), expected, atol=100.0)

    @pytest.mark.parametrize(
        "corpus, expected_exception",
        [
            (Corpus(), pytest.raises(ValueError)),
            (Corpus([Passage("test text")]), pytest.raises(ValueError)),
            (Corpus([Passage(f"test text {str(i)}") for i in range(10)]), None),
        ],
    )
    def test_compress_embeddings(self, corpus: Corpus, expected_exception):
        corpus.embeddings = np.random.rand(len(corpus), 768)

        with expected_exception or does_not_raise():
            compressed = corpus.compress_embeddings()

        if expected_exception is None:
            assert compressed.shape == (len(corpus), 2)

            np.testing.assert_array_equal(corpus.embeddings_2d, compressed)

            assert corpus.embeddings.shape == (len(corpus), 768)

            if len(corpus) == 1:
                # For single samples, the output should be zeros
                assert_equal(compressed, np.zeros((len(corpus), 2)))

            assert all(passage.embedding_compressed for passage in corpus.passages)

    def test_fit_umap(self, corpus):
        assert not Corpus._is_fitted(corpus.umap)

        with pytest.raises(AttributeError):
            corpus.umap.transform(np.random.rand(10, 768))

        corpus._fit_umap()

        assert Corpus._is_fitted(corpus.umap)

        np.testing.assert_allclose(
            corpus.umap.transform(np.random.rand(10, 768)),
            np.zeros((10, 2)),
            atol=100.0,
        )

    def test_fit_vectorizer(self, corpus):
        assert not Corpus._is_fitted(corpus.vectorizer)

        with pytest.raises(AttributeError):
            corpus.vectorizer.transform(Passage("test text"))

        corpus._fit_vectorizer()

        assert Corpus._is_fitted(corpus.vectorizer)
        np.testing.assert_equal(
            corpus.vectorizer.transform([Passage("test text")]).toarray(),
            np.array([[0.7071067811865475, 0.7071067811865475]]),
        )

        assert corpus.vectorizer.get_feature_names_out().tolist() == ["test", "text"]

    def test_tf_idf(self, corpus):
        assert not corpus._is_fitted(corpus.vectorizer)

        np.testing.assert_equal(
            corpus._tf_idf().toarray(),
            np.array([[0.7071067811865475, 0.7071067811865475]] * 5),
        )

        assert corpus._is_fitted(corpus.vectorizer)

    @pytest.mark.parametrize(
        "exclude_words, n, expected",
        [(None, 1, ["test"]), (None, 5, ["test", "text"]), (("test",), 1, ["text"])],
    )
    def test_top_words(self, corpus, exclude_words, n, expected):
        assert corpus.top_words(exclude_words=exclude_words, n=n) == expected
