import numpy as np
import csv


class OW:
    
    def __init__(self,csv_path = 'OutputCsv/output.csv'):
        self.csv_path = csv_path
    
    def write_numpy_array_to_csv(self, array):
        with open(self.csv_path, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(array)
    
    def append_numpy_array_to_csv(self, array):
        
        with open(self.csv_path, 'ab') as file:
            np.savetxt(file, np.matrix(array), delimiter=',')
        # array_as_list = array.tolist()
        # with open(self.csv_path, 'ab') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(array_as_list)