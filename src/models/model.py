import torch.nn as nn
from classy_vision.models import ClassyModel, register_model
import torch.nn.functional as F

@register_model("my_model")
class FCMaxPool(ClassyModel):
    def __init__(self, input_ch_size=9, grid_num=10, num_classes=2):
        super(FCMaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(input_ch_size * (grid_num//2) * (grid_num//2), 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.grid_num = grid_num
        self.input_ch_size = input_ch_size
        self.num_classes = num_classes
        
    @classmethod
    def from_config(cls, config):
        # This method takes a configuration dictionary 
        # and returns an instance of the class. In this case, 
        # we'll let the number of classes be configurable.
        return cls(num_classes=config["num_classes"],
                   grid_num=config["grid_num"],
                   input_ch_size=config["input_ch_size"])
    
    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, self.input_ch_size * (self.grid_num//2) * (self.grid_num//2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

