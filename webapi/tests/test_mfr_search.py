import unittest
from unittest.mock import patch

import core


class FakeColumn(list):
    def astype(self, target_type):
        return [target_type(value) for value in self]


class FakeDatasetExplorer:
    def get_file_list(self, group, where):
        self.assert_group(group)
        self.assert_label_filter(where)
        return [101, "202", 303]

    def get_file_info_all(self):
        return {
            "id": FakeColumn([101, 202]),
            "description": ["bracket_a.stp", "housing_b.stp"],
        }

    @staticmethod
    def assert_group(group):
        if group != "Labels":
            raise AssertionError(f"Unexpected group: {group}")

    @staticmethod
    def assert_label_filter(where):
        if not where({"face_labels": 18}):
            raise AssertionError("Expected feature name to resolve to face label 18.")
        if where({"face_labels": 24}):
            raise AssertionError("Unexpected match for a different face label.")


class MFRSearchTests(unittest.TestCase):
    def test_search_MFR_files_returns_file_names_and_list_for_feature_name(self):
        with patch.object(core, "get_MFR_dataset_explorer", return_value=FakeDatasetExplorer()):
            result = core.search_MFR_files("circular blind step")

        self.assertEqual(result["file_names"], ["bracket_a.stp", "housing_b.stp"])
        self.assertEqual(result["file_list"], [101, 202])


if __name__ == "__main__":
    unittest.main()
