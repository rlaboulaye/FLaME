import sys
import random
import json

from tqdm import tqdm
import numpy as np
import torch

from jsonschema import validate
from jsonschema.exceptions import ValidationError


def verbose_print(verbose, *args):
	if verbose:
		print(*args)

def get_iterator(obj, verbose=False):
	if verbose:
		return tqdm(obj, ncols=80)
	return iter(obj)

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def get_device(verbose=True):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if verbose:
		print("device: {}".format(device))
	return device

def validate_against_schema(schema_instance, schema_path):
	with open(schema_path, 'r') as schema_file:
		schema = json.load(schema_file)
	try:
		validate(schema_instance, schema)
	except ValidationError as err:
		sys.exit('EXCEPTION: THE SCHEMA INSTANCE FAILED TO VALIDATE AGAINST THE SCHEMA.\n\n{}'.format(err))