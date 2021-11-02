# we will be creating structured graphs of random lengths
import numpy as np

f = open("graphs.txt", "w")
f.write("")
f.close()
f = open("graphs.txt", "a")
stride = input("Give the stride: ")
stride = int(stride)
size = input("Give the max weight: ")
size = int(size)
string = ""
for x in range(stride):
    for y in range(stride):
        string = str(np.random.randint(size) + 1)
        if y < (stride - 1):
            string+=str(" ")
        else:
            string += str("\n")
        f.write(string)
