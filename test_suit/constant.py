from pathlib import Path

curr_path = Path(__file__).parents[1]
data_directory = curr_path/'test_suit/data'
pump_dataset_path_x = data_directory/'x_test.csv'
pump_dataset_path_y = data_directory/'y_test.csv'
