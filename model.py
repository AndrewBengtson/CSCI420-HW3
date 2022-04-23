"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, args):
		super(CNNModel, self).__init__()
		##-----------------------------------------------------------
		## define the model architecture here
		## MNIST image input size batch * 28 * 28 (one input channel)
		##-----------------------------------------------------------
		
		## define CNN layers below
		#sequential means it will automatically feed
		self.conv = nn.sequential( 	nn.Conv2d(1,10,2,2),
									nn.ReLu(),
									nn.Dropout(p=0.2),
									nn.Conv2d(10,8,2,2),
									nn.ReLu(),
									nn.Dropout(p=0.2),
									nn.Conv2d(8,6,2,2),
									nn.ReLu(),
									nn.Dropout(p=0.2),
									nn.Conv2d(6,4,2,2),

								)

		##------------------------------------------------
		## write code to define fully connected layer below
		##------------------------------------------------
		in_size = 4*28*28
		out_size = 10 #this is because we have 10 different categories we are attempting to predict
		self.fc = nn.Linear(in_size, out_size)
		

	'''feed features to the model'''
	def forward(self, x):  #default
		
		##---------------------------------------------------------
		## write code to feed input features to the CNN models defined above
		##---------------------------------------------------------
		x_out = self.conv(x)

		## write flatten tensor code below (it is done)
		x = torch.flatten(x_out,1) # x_out is output of last layer
		

		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result =   self.fc(x)
		
		
		return result
        
		
def calc_output_size(Width,Kernel,Padding,Stride):
	return ((Width-Kernel+(2*Padding))/Stride)+1
	
		