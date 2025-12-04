'''Checks how many fields have been completed in a directory. Updates a log file with the results.'''

import os
import numpy as np
import argparse

def check_completion(directory):
    directories = np.array(os.listdir(directory)).astype(int) # Fields are named as integers
    directories.sort()
    return directories

def write_log(log_file, completed_fields):
    with open(log_file, 'w') as f:
        for field in completed_fields:
            f.write(f"{field}\n")

# directory = '/data/zvar/zvar_results'
# log_file = 'completion_log.txt'
# completed_fields = check_completion(directory, log_file)
# write_log(log_file, completed_fields)

if __name__ == "__main__":
    #Take directory and log_file as inputs
    parser = argparse.ArgumentParser(description='Check completion of fields in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing field folders')
    parser.add_argument('log_file', type=str, help='File to write completion log')
    args = parser.parse_args()

    completed_fields = check_completion(args.directory)
    write_log(args.log_file, completed_fields)