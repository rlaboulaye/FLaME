import os
import json
import time
import datetime

from matplotlib import pyplot as plt


class Logger(object):

	def __init__(self, hyperparams, task_name, sequence_dim, scores_per_epoch=1):
		self.task_name = task_name + '_{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
		self.results_directory = os.path.join('results', self.task_name)
		self.results = {
			'train_losses': [],
			'validation_losses': [],
			'test_loss': 0,
			'scores_per_epoch': scores_per_epoch,
			'sequence_dim': sequence_dim,
			'hyperparams': hyperparams
		}

	def load(self, file_path):
		self.results_directory, task_file_name = os.path.split(file_path)
		self.task_name = os.path.splitext(task_file_name)[0]
		with open(file_path, 'r') as file_obj:
			self.results = json.load(file_obj)

	def log(self):
		if not os.path.exists(self.results_directory):
			os.makedirs(self.results_directory)
		with open('{}/results.json'.format(self.results_directory), 'w') as file_obj:
			json.dump(self.results, file_obj, indent=4)

	def plot(self):
		plt.figure()
		plt.title('Loss')
		plt.xlabel('1/{} Epoch'.format(self.results['scores_per_epoch']))
		plt.ylabel('Loss')
		plt.plot(self.results['train_losses'], label='train')
		plt.plot(self.results['validation_losses'], label='validate')
		plt.legend()
		plt.savefig('{}/loss.png'.format(self.results_directory))
		plt.close()
