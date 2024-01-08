from pathlib import Path

curr_path = Path(__file__).parents[1]
data_directory = curr_path / 'data'
x_independent_data_path = data_directory / 'x_pump_it_up.csv'
y_dependent_data_path = data_directory / 'y_pump_it_up.csv'
x_predict_data_path = data_directory / 'x_predict.csv'
