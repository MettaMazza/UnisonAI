import re
import unittest

from omni.pair_retrieval import PairRetrieval, _dialogue_act, _focus_words


class PairResponseBindingTests(unittest.TestCase):
    def test_focus_separates_operation_from_subject(self):
        self.assertEqual(_focus_words("tell me about space"), ["space"])
        self.assertEqual(_focus_words("what do you think about the ocean"), ["ocean"])
        self.assertEqual(_focus_words("recommend a good book"), ["book"])
        self.assertEqual(_focus_words("I just finished a painting"), ["painting"])

    def test_dialogue_act_is_preserved_without_final_punctuation(self):
        self.assertEqual(_dialogue_act("what makes a good friend"), "question")
        self.assertEqual(_dialogue_act("tell me about space"), "explain")
        self.assertEqual(_dialogue_act("recommend a good book"), "recommend")
        self.assertEqual(_dialogue_act("I finished a painting"), "statement")

    @staticmethod
    def _engine(prompt, response):
        engine = PairRetrieval()
        fingerprint = re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", prompt.lower())).strip()
        pairs = {
            "N": 1,
            "prompts": [prompt],
            "responses": [response],
            "exact": {fingerprint: [0]},
            "src_certain": [1],
        }
        engine._P = pairs
        engine._taught = []
        engine._quality = {}
        engine.retrieve = lambda text, history=None, topn=10: []
        engine._rarity = lambda word: 1.0
        return engine

    def test_assistant_self_seat_binds_to_unison_not_user(self):
        prompt = "My name is Maria, what is your name?"
        engine = self._engine(prompt, "Hi! I'm Sarah.")
        self.assertEqual(engine.reply(prompt), "Hi! I'm Unison.")

    def test_complete_response_unit_precedes_unrelated_splice(self):
        prompt = "I'm feeling sad."
        engine = self._engine(prompt, "Oh, I'm sorry to hear that. What's wrong?")
        self.assertEqual(engine.reply(prompt), "Oh, I'm sorry to hear that.")


if __name__ == "__main__":
    unittest.main()
