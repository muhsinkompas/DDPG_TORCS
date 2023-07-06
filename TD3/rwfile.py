import csv



class RW:
    
    def read_float_from_file(filename):
        if filename.endswith('.csv'):
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    float_value = float(row[0])  # Assuming the float value is in the first column
                    return float_value
        else:
            with open(filename, 'r') as file:
                content = file.read().strip()  # Read the entire file content as a string
                float_value = float(content)  # Convert the string to a float
                return float_value

    def write_float_to_file(filename, new_float_value):
        if filename.endswith('.csv'):
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([new_float_value])
        else:
            with open(filename, 'w') as file:
                file.write(str(new_float_value))

if __name__ == "__main__":
    # Usage example
    try:
        file_path = 'Best/bestlaptime.csv'  # Replace with the path to your file
        current_float_value = RW.read_float_from_file(file_path)
        print("Current float value:", current_float_value)
    except:
        print("file not existed")

    new_float_value = 9993.14  # Replace with the new float value you want to write
    RW.write_float_to_file(file_path, new_float_value)
    print("New float value:", new_float_value)
