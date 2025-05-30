import subprocess
import time
import os
import json

# Path to the directory containing ZoKrates programs
# program_directory = "./program"
program_directory = "./programs"
top_directory = "."

# Function to get the size of a file
def get_file_size(file_path):
    return os.path.getsize(file_path)

# Function to measure compilation time
def compile_program(program_path):
    start_time = time.time()
    compile_output = subprocess.run(["zokrates", "compile", "-i", program_path])
    end_time = time.time()
    output_file_path = os.path.join(top_directory, 'out')
    circuit_size = get_file_size(output_file_path)
    return end_time - start_time, circuit_size

# Function to execute the program and measure execution time
def execute_program(program_path):
    # input_path = "./input/bst1_instr.txt"
    start_time = time.time()
    subprocess.run(["zokrates", "compute-witness", "-a", '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '1', '10000000000', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '10000000000', '1', '0', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '10000000000', '1', '200000000000', '1', '200000000000', '1', '200000000000', '1', '200000000000', '1', '200000000000', '1', '10000000000', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '10000000000', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '1', '10000000000', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '1', '0', '1', '10000000000', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '10000000000', '1', '480000000000', '1', '480000000000', '1', '480000000000', '1', '480000000000', '1', '480000000000', '1', '0', '1', '10000000000', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '10000000000', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '0', '999990000000000', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '10000000000', '1', '0', '1', '0', '1', '0', '1', '0'])
    return time.time() - start_time

# Function to generate proof and extract proof size and number of constraints
def generate_proof(program_path):
    subprocess.run(["zokrates", "setup"])
    subprocess.run(["zokrates", "export-verifier"])
    start_time = time.time()
    proof_output = subprocess.run(["zokrates", "generate-proof"])
    end_time = time.time()
    # Example of extracting proof size and constraints from the output
    # This depends on the format of the output from ZoKrates
    output_file_path = os.path.join(top_directory, 'proof.json')
    proof_size = get_file_size(output_file_path)
    
    return proof_size

def verify_proof(program_path):
    start_time = time.time()
    subprocess.run(["zokrates", "verify"])
    end_time = time.time()
    
    return end_time - start_time


# Main function to batch test ZoKrates programs
def batch_test_zokrates_programs():
    results = []

    for program in os.listdir(program_directory):
        # Skip hidden files and directories
        if program.startswith('.') or os.path.isdir(os.path.join(program_directory, program)):
            continue

        program_path = os.path.join(program_directory, program)
        compile_time, circuit_size = compile_program(program_path)
        execution_time = execute_program(program_path)
        proof_size = generate_proof(program_path)
        verification_time = verify_proof(program_path)

        results.append({
            "program": program,
            "compile_time": compile_time,
            "circuit_size": circuit_size,
            "execution_time": execution_time,
            "proof_size": proof_size,
            "verification_time": verification_time
        })

    # Write results to a JSON file
    with open('zokrates_test_results.json', 'w') as file:
        json.dump(results, file, indent=4)

batch_test_zokrates_programs()