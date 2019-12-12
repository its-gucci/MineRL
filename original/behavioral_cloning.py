import torch 

class BehavioralCloning():
	def __init__(self, model, optimizer, batch_size):
		self.model = model
		self.opt = optimizer
		self.batch_size = batch_size

		self.average_loss = 0

    def _loss(self, x, labels):
    	logits = self.model(x)
    	return 

	def train(self):
		return

	def get_statistics(self):
		return {'average_loss': self.average_loss}
