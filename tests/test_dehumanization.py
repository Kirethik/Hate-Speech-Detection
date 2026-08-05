"""Tests for the dehumanisation lexicon."""

from dehumanization import explain, find_frames, is_dehumanizing_claim, score_text


class TestFrameDetection:
    def test_detects_disease_frame(self):
        assert "disease" in find_frames("they are a plague on this town")

    def test_detects_vermin_frame(self):
        assert "vermin" in find_frames("these immigrants are vermin")

    def test_detects_animal_frame(self):
        assert "animal" in find_frames("they are subhuman apes")

    def test_detects_multiple_frames(self):
        frames = find_frames("they are filthy vermin spreading disease")
        assert {"filth", "vermin", "disease"} <= set(frames)

    def test_returns_empty_for_clean_text(self):
        assert find_frames("the soundtrack was beautiful") == {}

    def test_is_case_insensitive(self):
        assert "disease" in find_frames("They Are A PLAGUE")


class TestMotivatingFailureCase:
    """The exact sentence the trained model scored 0.28 (clean)."""

    TEXT = ("The Bronzites are a plague on every town they enter. "
            "Nothing good has ever come from trusting one of them.")

    def test_lexicon_catches_what_the_model_missed(self):
        assert is_dehumanizing_claim(self.TEXT)

    def test_identifies_the_disease_frame(self):
        assert score_text(self.TEXT)["frames"] == ["disease"]

    def test_works_for_a_group_name_never_seen_in_training(self):
        # The whole point: "Bronzites" is invented, so no lookup table helps.
        assert "bronzite" not in str(find_frames(self.TEXT)).lower()
        assert is_dehumanizing_claim(self.TEXT)

    def test_explanation_is_human_readable(self):
        assert "disease frame" in explain(self.TEXT)
        assert "plague" in explain(self.TEXT)


class TestRequiresGroupSubject:
    """A lexicon without a subject check is just keyword matching."""

    def test_disease_about_an_actual_disease_is_not_a_claim(self):
        assert not is_dehumanizing_claim("the flu is a plague this winter")

    def test_object_complaint_is_not_a_claim(self):
        assert not is_dehumanizing_claim("this game is cancer")

    def test_frame_still_reported_even_when_not_a_claim(self):
        s = score_text("this game is cancer")
        assert s["frames"] == ["disease"]
        assert not s["is_dehumanizing_claim"]

    def test_explanation_says_why_it_did_not_fire(self):
        assert "not predicated of a group" in explain("this game is cancer")

    def test_pronoun_subject_counts_as_a_group(self):
        assert is_dehumanizing_claim("they are vermin and should be removed")


class TestNoFalsePositivesOnBenignText:
    def test_plain_benign(self):
        assert not is_dehumanizing_claim("i really enjoyed this movie")

    def test_identity_mention_without_metaphor(self):
        assert not is_dehumanizing_claim(
            "i am a muslim woman and i am proud of my heritage"
        )

    def test_activism_without_metaphor(self):
        assert not is_dehumanizing_claim("black lives matter and we will keep marching")

    def test_empty_string(self):
        assert not is_dehumanizing_claim("")

    def test_explanation_empty_when_nothing_fires(self):
        assert explain("the weather is nice") == ""
