import sys
import re

# Define the functions for string parsing

def get_ctests(ctest_output):

    # Extract the executable path

    ctests = []
    lines = ctest_output.split("_NEWLINE_")
    index = 0

    while index < len(lines):
        line = lines[index]
        index += 1

        CmdStr = ": Test command: "
        posCmd = line.find(CmdStr)
        if posCmd < 0:
            continue

        Cmd = line[posCmd+len(CmdStr):]
        Exe = Cmd.split()[-1]

        line = lines[index]
        index += 1
        
        DirStr = ": Working Directory:"
        posDir = line.find(DirStr)
        if posDir < 0:
            continue

        Dir = line[posDir+len(DirStr):]
 
        line = lines[index]
        index += 1
        
        TestStr = "Test"
        posTest = line.find(TestStr)
        if posTest < 0:
            continue

        Test = line.split()[-1]
        ctests.append(f"{Cmd},{Exe},{Dir},{Test}")
    
    return ";".join(ctests)

def reverse_words(input_str):
    """Reverse the order of words in a string."""
    words = input_str.split()
    return ";".join(words[::-1])

def to_upper(input_str):
    """Convert all words in a string to uppercase."""
    words = input_str.split()
    return ";".join([word.upper() for word in words])

def remove_vowels(input_str):
    """Remove vowels from the string."""
    vowels = "aeiouAEIOU"
    return ";".join([char for char in input_str if char not in vowels])

# Main script to handle command-line arguments and execute the correct function

def main():

    if len(sys.argv) < 3:
        return ""

    function_name = sys.argv[1]
    input_argument = " ".join(sys.argv[2:])

    # Try to get the function by name from the globals()
    try:
        func = globals()[function_name]
        # Ensure the function is callable
        if not callable(func):
            raise ValueError(f"{function_name} is not callable")
    except KeyError:
        raise ValueError(f"No function named '{function_name}' found.")
    
    # Call the selected function with the input argument
    result = func(input_argument)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()

