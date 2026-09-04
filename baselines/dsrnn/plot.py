import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


legends = ['DSRNN', '']

# add more training curves by directory name here!
log_list = [pd.read_csv("data/trained/progress.csv"),
		   ]

logDicts = {}
for i in range(len(log_list)):
	logDicts[i] = log_list[i]

graphDicts={0:'eprewmean', 1:'loss/value_loss'}

legendList=[]
# summarize history for accuracy

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0
	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.plot(logDicts[key]['misc/total_timesteps'],logDicts[key][graphDicts[i]])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='lower right')
	legendList=[]

	# Save inside the loop, once per metric's own figure — savefig() outside
	# the loop only ever captures whichever figure was created last (here,
	# loss/value_loss), silently dropping the reward figure instead of
	# writing it to disk.
	metric_slug = graphDicts[i].replace('/', '_')
	plt.savefig(f'HEIGHTtraining_curves_{metric_slug}.png')

plt.show()


