import json
import unittest
from pathlib import Path

from scene_understanding.core.vla_action_proposal import (
    EXPECTED_FIELDS,
    VLA_ACTIONS,
    VLA_ACTION_PROPOSAL_SCHEMA_VERSION,
    VLA_PROPOSAL_STATUSES,
    validate_vla_action_proposal,
)


class VlaActionProposalContractTests(
    unittest.TestCase
):
    def _scene_root(self):
        return Path(__file__).resolve().parents[2]

    def _read_json(self, relative_path):
        path = (
            self._scene_root()
            / relative_path
        )
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def test_checked_in_example_is_valid(self):
        example = self._read_json(
            "schemas/examples/"
            "vla_action_proposal.example.json"
        )

        self.assertEqual(
            validate_vla_action_proposal(
                example
            ),
            [],
        )

    def test_schema_version_matches_code(self):
        schema = self._read_json(
            "schemas/"
            "vla_action_proposal.schema.json"
        )

        self.assertEqual(
            schema["properties"][
                "schema_version"
            ]["const"],
            VLA_ACTION_PROPOSAL_SCHEMA_VERSION,
        )

    def test_required_fields_match_code(self):
        schema = self._read_json(
            "schemas/"
            "vla_action_proposal.schema.json"
        )

        self.assertEqual(
            set(schema["required"]),
            EXPECTED_FIELDS,
        )
        self.assertFalse(
            schema["additionalProperties"]
        )

    def test_action_enum_matches_code(self):
        schema = self._read_json(
            "schemas/"
            "vla_action_proposal.schema.json"
        )

        self.assertEqual(
            set(
                schema["properties"][
                    "action"
                ]["enum"]
            ),
            VLA_ACTIONS,
        )

    def test_status_enum_matches_code(self):
        schema = self._read_json(
            "schemas/"
            "vla_action_proposal.schema.json"
        )

        self.assertEqual(
            set(
                schema["properties"][
                    "status"
                ]["enum"]
            ),
            VLA_PROPOSAL_STATUSES,
        )


if __name__ == "__main__":
    unittest.main()
