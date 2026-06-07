import pytest

from mlx_cpt_plus import CPTPlus, Config


def test_model_fit_and_predict(sample_sequences):
    model = CPTPlus(config=Config(min_frequency=1, max_vocabulary_size=100))
    model.fit(sample_sequences)
    predictions = model.predict([1, 2, 3], k=3)

    assert isinstance(predictions, list)
    assert len(predictions) <= 3
    assert all(isinstance(item, tuple) and len(item) == 2 for item in predictions)


def test_sequence_store_count(sample_sequences):
    from mlx_cpt_plus.core.sequence_store import SequenceStore

    store = SequenceStore()
    for seq in sample_sequences:
        store.add(seq)

    assert len(store) == len(sample_sequences)
    assert store.count() == len(sample_sequences)
    assert store.get(0) == sample_sequences[0]
