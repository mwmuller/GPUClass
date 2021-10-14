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

for i in range(10):
	datasetDir = "Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/TiledMatrixMultiplication","-i",datasetDir+"input0.raw," + datasetDir+"input1.raw","-e",datasetDir+"output.raw","-t","matrix"])
	correct = re.findall('"correctq": true',result)
	aDimMatch = re.search('"The dimensions of A are (\\d+) x (\\d+)"',result)
	bDimMatch = re.search('"The dimensions of B are (\\d+) x (\\d+)"',result)

	totalFLOPS = 2 * int(aDimMatch.group(1)) * int(aDimMatch.group(2)) * int(bDimMatch.group(2))	
	tiledMatch = re.search('\\{.*"elapsed_time": (\\d+).*"message": "Performing basic tiled computation"',result)
	tiledTime = int(tiledMatch.group(1)) / 1e9

	print "basic tiling: " + str(totalFLOPS) + " total FLOPs / " + str(tiledTime) + " s = " + str(totalFLOPS / tiledTime / 1e9) + " GFLOPS / s"

	multitiledMatch = re.search('\\{.*"elapsed_time": (\\d+).*"message": "Performing multi-tiled computation"',result)
	multitiledTime = int(multitiledMatch.group(1)) / 1e9
	print "multi-tiling: " + str(totalFLOPS) + " total FLOPs / " + str(multitiledTime) + " s = " + str(totalFLOPS / multitiledTime / 1e9) + " GFLOPS / s"

	if len(correct)==2:
		nCorrect += 1
	else:
		print str(i) + " incorrect"

print str(nCorrect) + " / 10 correct"
