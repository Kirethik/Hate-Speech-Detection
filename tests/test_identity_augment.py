"""Tests for identity-term augmentation."""

import pytest

from identity_augment import (
    NONCE_GROUPS,
    IDENTITY_TERMS,
    IdentityAugmenter,
    contains_identity_term,
)


class TestContainsIdentityTerm:
    def test_detects_single_word_term(self):
        assert contains_identity_term("muslims are people too")

    def test_detects_multi_word_phrase(self):
        assert contains_identity_term("black people deserve respect")

    def test_is_case_insensitive(self):
        assert contains_identity_term("Muslims are people too")

    def test_returns_false_when_no_identity_term(self):
        assert not contains_identity_term("the weather is nice today")

    def test_respects_word_boundaries(self):
        # "asians" must not fire inside "caucasians"
        assert not contains_identity_term("caucasians")


class TestNonceSubstitution:
    def test_replaces_identity_term_with_invented_group(self):
        aug = IdentityAugmenter(swap_prob=0.0, nonce_prob=1.0, seed=1)
        text, _ = aug("muslims are vermin")
        assert not any(w in text.lower() for w in ["muslim", "muslims"])
        assert any(n.lower() in text.lower() for n in NONCE_GROUPS)

    def test_masks_target_label_because_nonce_has_no_real_category(self):
        aug = IdentityAugmenter(swap_prob=0.0, nonce_prob=1.0, seed=1)
        _, target = aug("muslims are vermin", target_label=1)
        assert target == -1

    def test_preserves_the_predicate(self):
        aug = IdentityAugmenter(swap_prob=0.0, nonce_prob=1.0, seed=1)
        text, _ = aug("muslims are a plague on every town")
        assert "are a plague on every town" in text

    def test_bronzites_is_never_used_since_it_is_the_probe_case(self):
        assert not any("bronzite" in n.lower() for n in NONCE_GROUPS)


class TestWithinCategorySwap:
    def test_replaces_term_with_different_term(self):
        aug = IdentityAugmenter(swap_prob=1.0, nonce_prob=0.0, seed=3)
        text, _ = aug("muslims are vermin")
        assert "muslims" not in text.lower()

    def test_keeps_target_label_since_category_is_unchanged(self):
        aug = IdentityAugmenter(swap_prob=1.0, nonce_prob=0.0, seed=3)
        _, target = aug("muslims are vermin", target_label=1)
        assert target == 1

    def test_replacement_stays_within_the_same_category(self):
        aug = IdentityAugmenter(swap_prob=1.0, nonce_prob=0.0, seed=7)
        religion = {t.lower() for t in IDENTITY_TERMS["religion"]}
        for _ in range(25):
            text, _ = aug("muslims should be deported")
            replaced = text.lower().replace(" should be deported", "")
            assert replaced in religion


class TestPassthrough:
    def test_leaves_text_without_identity_terms_untouched(self):
        aug = IdentityAugmenter(swap_prob=1.0, nonce_prob=0.0, seed=1)
        text, target = aug("the soundtrack was beautiful", target_label=0)
        assert text == "the soundtrack was beautiful"
        assert target == 0

    def test_zero_probability_disables_augmentation(self):
        aug = IdentityAugmenter(swap_prob=0.0, nonce_prob=0.0, seed=1)
        text, target = aug("muslims are vermin", target_label=1)
        assert text == "muslims are vermin"
        assert target == 1

    def test_empty_text_is_safe(self):
        aug = IdentityAugmenter(swap_prob=1.0, nonce_prob=0.0, seed=1)
        assert aug("") == ("", -1)


class TestCasePreservation:
    def test_capitalised_term_yields_capitalised_replacement(self):
        aug = IdentityAugmenter(swap_prob=0.0, nonce_prob=1.0, seed=1)
        text, _ = aug("Muslims are vermin")
        assert text[0].isupper()


class TestConfiguration:
    def test_rejects_probabilities_summing_above_one(self):
        with pytest.raises(ValueError):
            IdentityAugmenter(swap_prob=0.7, nonce_prob=0.7)

    def test_is_deterministic_for_a_given_seed(self):
        a = IdentityAugmenter(swap_prob=0.5, nonce_prob=0.3, seed=99)
        b = IdentityAugmenter(swap_prob=0.5, nonce_prob=0.3, seed=99)
        texts = ["muslims are vermin", "women are stupid", "immigrants ruin everything"]
        assert [a(t) for t in texts] == [b(t) for t in texts]
