import torch

from respirazione.training import LengthBucketBatchSampler

LENGTHS = [500, 80, 1090, 300, 620, 150, 940, 210, 760, 430]


def test_every_index_appears_exactly_once():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=3, shuffle=False)
    indici = [i for batch in sampler for i in batch]
    assert sorted(indici) == list(range(len(LENGTHS)))


def test_len_matches_the_batches_produced():
    for batch_size in (1, 3, 4, len(LENGTHS), len(LENGTHS) + 5):
        sampler = LengthBucketBatchSampler(LENGTHS, batch_size=batch_size)
        assert len(list(iter(sampler))) == len(sampler)


def test_batches_group_similar_lengths():
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=3, shuffle=False)
    # senza raggruppamento il padding sarebbe la lunghezza massima in ogni batch
    padding_bucket = sum(len(b) * max(LENGTHS[i] for i in b) for b in sampler)
    padding_peggiore = len(LENGTHS) * max(LENGTHS)
    assert padding_bucket < padding_peggiore


def test_shuffle_changes_the_order_but_not_the_content():
    torch.manual_seed(0)
    sampler = LengthBucketBatchSampler(LENGTHS, batch_size=3, shuffle=True)
    primo, secondo = list(iter(sampler)), list(iter(sampler))
    assert sorted(i for b in primo for i in b) == sorted(i for b in secondo for i in b)
