import subprocess
import re

if "check_output" not in dir( subprocess ):
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output
    subprocess.check_output = f
	
nCorrect = 0

for i in range(7):
	datasetDir = "./Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/Convolution","-i",datasetDir+"input0.ppm,"+ datasetDir+"input1.raw","-e",datasetDir+"output.ppm","-t","image"])
	correct = re.search('"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 7 correct"
