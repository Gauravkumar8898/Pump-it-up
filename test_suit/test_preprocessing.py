import unittest
from constant import pump_dataset_path_x, pump_dataset_path_y
from src.pump_it_up_preprocessing.preprocessing import load_dataset, drop_unused_feature


class TestPreprocessing(unittest.TestCase):

    # In this function we have test the funtion loading and feature dropping
    def test_load_dataset_and_drop_features(self):
        test_dataset = load_dataset(pump_dataset_path_x)
        self.assertIsNotNone(test_dataset)
        test_dataset1 = drop_unused_feature(test_dataset)
        assert len(test_dataset1.columns) < len(test_dataset.columns)


if __name__ == '__main__':
    unittest.main()
