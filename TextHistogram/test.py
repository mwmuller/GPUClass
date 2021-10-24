import subprocess
import re

nCorrect = 0

for i in range(6):
	datasetDir = "./Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/TextHistogram","-i",datasetDir+"input.txt","-e",datasetDir+"output.raw","-t","integral_vector"])
	correct = re.search('"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 6 correct"
