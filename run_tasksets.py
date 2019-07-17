import subprocess
import pathos.pools as pp
import matplotlib.pyplot as plt

# print(process)
def job(i,thread):
	print("process number ",i)
	process = subprocess.Popen(['taskset',
								'-c',
								str(i),
								'python',
								'softlearning/map3D/main_parallel.py',str(thread)] )
	return process



def main():
	# p = pp.ProcessPool(4)
	# jobs = sorted(list(enumerate(enum_obj_paths())))
	# jobs = [(1,1),(1,1),(1,1)]
	# jobs = list(range(10))
	# st()
	# print(jobs)
	# job(jobs[0])
	timeVals = []
	# threads = list(range(41,50,2))
	threads= [19]
	threads.reverse()
	for thread in threads:
		print("number of threads",thread)
		import time 
		t = time.time()
		proc = []
		for i in range(0,40):
			proc.append(job(i,thread))
		# p.map(job, jobs, chunksize = 1)
		exit_codes = [i.wait() for i in proc]
		timeVal = time.time()-t
		print(time.time()-t,"total time",exit_codes)
		timeVals.append(timeVal)
	# threads
	# timeVals
	plt.plot(threads, timeVals, '-gD')
	plt.xlabel("Threads each Core")
	plt.ylabel("Time taken")
	import pickle
	pickle.dump([threads,timeVals],open("range41-50.p","wb"))
	plt.savefig("threadsvstime.png")   	


if __name__ == "__main__":
	main()