import os
import json

from matplotlib import pyplot as plt


class Logger(object):

	def __init__(self, config, task_name, scores_per_epoch=1, default_accuracy=.5):
		self.task_name = task_name
		self.results_directory = os.path.join('results', self.task_name)
		self.results = {
			'train_losses': [],
			'train_accuracies': [],
			'validation_losses': [],
			'validation_accuracies': [],
			'test_loss': 0,
			'test_accuracy': 0,
			'scores_per_epoch': scores_per_epoch,
			'default_accuracy': default_accuracy,
			'config': config
		}
		if not os.path.exists(self.results_directory):
			os.makedirs(self.results_directory)

	def load(self, file_path):
		self.results_directory, task_file_name = os.path.split(file_path)
		self.task_name = os.path.splitext(task_file_name)[0]
		with open(file_path, 'r') as file_obj:
			self.results = json.load(file_obj)

	def log(self):
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

		plt.figure()
		plt.title('Accuracy')
		plt.xlabel('1/{} Epoch'.format(self.results['scores_per_epoch']))
		plt.ylabel('Accuracy')
		plt.plot(self.results['train_accuracies'], label='train')
		plt.plot(self.results['validation_accuracies'], label='validate')
		plt.plot([self.results['default_accuracy']] * len(self.results['train_accuracies']), label='default')
		plt.legend()
		plt.savefig('{}/accuracy.png'.format(self.results_directory))
		plt.close()
