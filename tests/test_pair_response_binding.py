import re
import unittest

from omni.pair_retrieval import PairRetrieval, _dialogue_act, _focus_words, _shape_response


class PairResponseBindingTests(unittest.TestCase):
    def test_focus_separates_operation_from_subject(self):
        self.assertEqual(_focus_words("tell me about space"), ["space"])
        self.assertEqual(_focus_words("what do you think about the ocean"), ["ocean"])
        self.assertEqual(_focus_words("recommend a good book"), ["book"])
        self.assertEqual(_focus_words("I just finished a painting"), ["painting"])
        self.assertEqual(_focus_words("I'm Maria. What should I call you?"), ["name"])
        self.assertEqual(_focus_words("Give me a simple overview of outer space."), ["space"])
        self.assertEqual(_focus_words("Any ideas for a vegetarian supper?"),
                         ["vegetarian", "meal"])
        self.assertEqual(_focus_words("I completed a watercolor today."),
                         ["watercolor", "painting"])
        self.assertEqual(_focus_words("How do you spend your free time?"), ["hobbies"])
        self.assertEqual(_focus_words("what makes a good friend"),
                         ["qualities", "friend"])

    def test_dialogue_act_is_preserved_without_final_punctuation(self):
        self.assertEqual(_dialogue_act("what makes a good friend"), "criteria")
        self.assertEqual(_dialogue_act("tell me about space"), "explain")
        self.assertEqual(_dialogue_act("recommend a good book"), "recommend")
        self.assertEqual(_dialogue_act("I finished a painting"), "statement")
        self.assertEqual(_dialogue_act("Give me an overview of space"), "explain")
        self.assertEqual(_dialogue_act("Which qualities matter in friendship?"),
                         "criteria")

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
        self.assertEqual(engine.reply(prompt), "I'm Unison.")
        self.assertEqual(engine.reply("I'm Maria. What should I call you?"),
                         "I'm Unison.")

    def test_complete_response_unit_precedes_unrelated_splice(self):
        prompt = "I'm feeling sad."
        engine = self._engine(prompt, "Oh, I'm sorry to hear that. What's wrong?")
        self.assertEqual(engine.reply(prompt), "Oh, I'm sorry to hear that.")

    def test_explicit_one_sentence_request_selects_leading_closed_unit(self):
        prompt = "Explain space exploration in one sentence."
        response = ("Space exploration reaches beyond Earth's atmosphere. "
                    "Its history includes many developments.")
        engine = self._engine(prompt, response)
        self.assertEqual(engine.reply(prompt),
                         ("Space exploration reaches beyond Earth's atmosphere; "
                          "its history includes many developments."))
        self.assertEqual(
            _shape_response(prompt, response),
            "Space exploration reaches beyond Earth's atmosphere; "
            "its history includes many developments.")

    def test_coordinated_clause_is_a_closed_nonverbatim_unit(self):
        prompt = "What causes ocean tides?"
        response = ("Tides are very long waves moving across the ocean and are caused "
                    "by the gravitational forces of the moon.")
        engine = self._engine(prompt, response)
        self.assertEqual(engine.reply(prompt),
                         "Tides are very long waves moving across the ocean.")

    def test_user_name_recall_reads_append_only_history(self):
        engine = self._engine("What did I tell you my name was?", "unused")
        self.assertEqual(engine.reply(
            "What did I tell you my name was?",
            history=[("user", "My name is Maria.")]),
            "You told me your name was Maria.")
        self.assertEqual(engine.reply("What did I tell you my name was?", history=[]), "")

    def test_mutual_kin_value_cannot_admit_a_one_way_relation(self):
        engine = PairRetrieval()
        bands = {"sea": frozenset({"ocean", "rising"}),
                 "ocean": frozenset({"sea"}),
                 "rising": frozenset()}
        engine._kin_top = lambda word: bands.get(word, frozenset())
        engine._kin_store = {"sea": {"ocean": 0.4, "rising": 0.3},
                             "ocean": {"sea": 0.4}}
        self.assertEqual(engine._surface_credit("sea", "ocean"), 0.4)
        self.assertEqual(engine._surface_credit("sea", "rising"), 0.0)


if __name__ == "__main__":
    unittest.main()
