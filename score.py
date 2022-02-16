import sys

# A program to compute the accuracy of your system based on its output
#
# Usage:
#
# $ python score.py OUTPUT_FILE KEYS_FILE
spaceCHAR = [0]
output_fname = sys.argv[1]
keys_fname = sys.argv[2]
# output = []

# def getASCII(x):
#     ascii_values = []
#     for character in x.strip():
#         ascii_values.append(ord(character))
#     return ascii_values
# for x in open(output_fname):
#     ascii_val = getASCII(x)
#     if(ascii_val != spaceCHAR):
#         output.append(x)
#         x = x.strip()
#         print(x)
# Read in the system output and keys
output = [ x.strip() for x in open(output_fname)]
keys = [x.strip() for x in open(keys_fname)]
print(len(output))
if len(output) != len(keys):
    # If the length of the output and keys are not the same, something went
    # wrong...
    print("Invalid output: Incorrect number of lines")
else:
    num_correct = 0
    total = 0
    
    for o,k in zip(output,keys):
    
        if o == k:
            num_correct += 1
        total += 1
    accuracy = num_correct / total
    print("Num correct: ", num_correct)
    print("Total: ", total)
    print("Accuracy:", round(accuracy, 3))
