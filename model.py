"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, args):
		super(CNNModel, self).__init__()
		##--------------------------------------------------------
		## define the model architecture here
		## image input size batch * 28 * 28
		##--------------------------------------------------------
		
		## define CNN layers below
		
		
		## define fully connected layer below
		self.fc = nn.Linear(in_size, out_size)
		

	'''feed features to the model'''
	def forward(self, x):
		##---------------------------------------------------
		## feed input features to the models defined above
		##---------------------------------------------------
		

		## write flatten tensor code below
		x = torch.flatten(x,1)

		## --------------------------------------------------
		## write fully connected layer (Linear layer) below
		## --------------------------------------------------
		


		return results
        
		
		
		

		