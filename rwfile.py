import csv
import numpy as np
import os.path


class RW:

    def __init__(self,csv_path = 'Best/bestlaptime.csv'):
        self.csv_path = csv_path
        if os.path.isfile(self.csv_path) == False:
            print("File doesn't exist.")
            self.write_numpy_array_to_csv(np.array((9993.14)))
            
    
    def write_numpy_array_to_csv(self, array):
        with open(self.csv_path, 'wb') as file:
            np.savetxt(file, np.matrix(array), delimiter=',')
            #writer = csv.writer(csvfile)
            #writer.writerows(array)
    
    def append_numpy_array_to_csv(self, array):
        
        with open(self.csv_path, 'ab') as file:
            np.savetxt(file, np.matrix(array), delimiter=',')
            
    def read_numpy_array_from_csv(self):
            
            float_value = np.loadtxt(self.csv_path , delimiter=',',dtype=float)
            return float_value
            
            
    # def read_float_from_file(self,filename):
    #     #data = np.genfromtxt(filename)
    #     if filename.endswith('.csv'):
    #         with open(filename, 'rb') as file:
    #             reader = csv.reader(file)
    #             for row in reader:
    #                 float_value = float(row[0])  # Assuming the float value is in the first column
    #                 return float_value
    #     else:
    #         with open(filename, 'rb') as file:
    #             content = file.read().strip()  # Read the entire file content as a string
    #             float_value = float(content)  # Convert the string to a float
    #             return float_value

    # def write_float_to_file(self, filename, new_float_value):
    #     if filename.endswith('.csv'):
    #         with open(filename, 'wb') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([new_float_value])
    #     else:
    #         with open(filename, 'wb') as file:
    #             file.write(str(new_float_value))
                
if __name__ == u"__main__":
    r_w = RW()
    
    # Usage example
    file_path = 'Best/bestlaptime.csv'  # Replace with the path to your file
    current_float_value = r_w.read_numpy_array_from_csv()
    print("Current float value:", current_float_value)

    # new_float_value = 3.14  # Replace with the new float value you want to write
    # r_w.write_float_to_file(file_path, new_float_value)
    # print("New float value:", new_float_value)

