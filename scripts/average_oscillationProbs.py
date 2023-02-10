import numpy as np
import glob
import sys

def average_files(files):
    data = []
    completed = 0
    for file in files:
        data.append(np.loadtxt(file))
	completed += 1
        print("loaded " + str(completed) + " files out of " + str(len(files)))
    average = np.mean(data, axis=0)
    return average

def main():
    input_common = sys.argv[1]
    output = sys.argv[2]
    files = glob.glob(input_common + "*.txt")
    average = average_files(files)
    np.savetxt(output, average, delimiter="\t")

if __name__ == '__main__':
    main()
