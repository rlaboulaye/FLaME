import os
import json
import time
import datetime

from matplotlib import pyplot as plt
import torch


class Logger(object):

	def __init__(self, hyperparams, task_name, data_path):
		self.task_name = task_name + '_{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
		self.results_directory = os.path.join('results', self.task_name)
		self.params_directory = os.path.join('params', self.task_name)
		self.validation_frequency = hyperparams['validation_frequency']
		self.results = {
			'train_losses': {},
			'validation_losses': {},
			'test_loss': {},
			'data_path': data_path,
			'hyperparams': hyperparams
		}

	def load(self, file_path):
		self.results_directory, task_file_name = os.path.split(file_path)
		self.task_name = os.path.splitext(task_file_name)[0]
		with open(file_path, 'r') as file_obj:
			self.results = json.load(file_obj)

	def add_train_val_losses(self, train_losses, validation_losses):
		for label in train_losses:
			if not label in validation_losses:
				raise ValueError('Label {} should be in validation_losses'.format(label))
			if not label in self.results['train_losses']:
				self.results['train_losses'][label] = []
				self.results['validation_losses'][label] = []
			self.results['train_losses'][label].extend(train_losses[label])
			self.results['validation_losses'][label].extend(validation_losses[label])

	def set_test_losses(self, test_losses):
		self.results['test_loss'] = test_losses

	def log_results(self):
		if not os.path.exists(self.results_directory):
			os.makedirs(self.results_directory)
		with open('{}/results.json'.format(self.results_directory), 'w') as file_obj:
			json.dump(self.results, file_obj, indent=4)

	def log_weights(self, state_dict, name):
		if not os.path.exists(self.params_directory):
			os.makedirs(self.params_directory)
		torch.save(state_dict, os.path.join(self.params_directory, name))

	def plot(self):
		for label in self.results['train_losses']:
			train_losses = self.results['train_losses'][label]
			validation_losses = self.results['validation_losses'][label]
			self._plot(label, train_losses, validation_losses)

	def _plot(self, label, train_losses, validation_losses):
		plt.figure()
		plt.title('{} Loss'.format(label).replace('_', ' ').title())
		plt.xlabel('Batches')
		plt.ylabel('Loss')
		plt.plot(range(self.validation_frequency, self.validation_frequency * (1 + len(train_losses)),
			self.validation_frequency), train_losses, label='train')
		plt.plot(range(self.validation_frequency, self.validation_frequency * (1 + len(validation_losses)),
			self.validation_frequency), validation_losses, label='validate')
		plt.legend()
		plt.savefig('{}/{}_loss.png'.format(self.results_directory, label))
		plt.close()
