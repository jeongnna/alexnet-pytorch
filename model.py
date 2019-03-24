from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class AlexNet(nn.Sequential):
    def __init__(self):
        super().__init__()
        
        # layer 1: convolutional layer
        self.add_module("layer1_conv", nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4))
        self.add_module("layer1_relu", nn.ReLU())
        self.add_module("layer1_norm", nn.LocalResponseNorm(size = 5, alpha = 0.0001 * 5, beta = 0.75, k = 2.0))
        self.add_module("layer1_pool", nn.MaxPool2d(kernel_size = 3, stride = 2))
        # layer 2: convolutional layer
        self.add_module("layer2_conv", nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2))
        self.add_module("layer2_relu", nn.ReLU())
        self.add_module("layer2_norm", nn.LocalResponseNorm(size = 5, alpha = 0.0001 * 5, beta = 0.75, k = 2.0))
        self.add_module("layer2_pool", nn.MaxPool2d(kernel_size = 3, stride = 2))
        
        # layer 3: convolutional layer
        self.add_module("layer3_conv", nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1))
        self.add_module("layer3_relu", nn.ReLU())
        
        # layer 4: convolutional layer
        self.add_module("layer4_conv", nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1))
        self.add_module("layer4_relu", nn.ReLU())
        
        # layer 5: convolutional layer
        self.add_module("layer5_conv", nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1))
        self.add_module("layer5_relu", nn.ReLU())
        self.add_module("layer5_pool", nn.MaxPool2d(kernel_size = 3, stride = 2))
        
        # layer 6: fully connected layer
        self.add_module("layer6_flatten", Flatten())
        self.add_module("layer6_linear", nn.Linear(in_features = 6 * 6 * 256, out_features = 4096))
        self.add_module("layer6_relu", nn.ReLU())
        self.add_module("layer6_dropout", nn.Dropout(p = 0.5))
        
        # layer 7: fully connected layer
        self.add_module("layer7_linear", nn.Linear(in_features = 4096, out_features = 4096))
        self.add_module("layer7_relu", nn.ReLU())
        self.add_module("layer7_dropout", nn.Dropout(p = 0.5))
        
        # layer 8: fully connected layer
        self.add_module("layer8_linear", nn.Linear(in_features = 4096, out_features = 1000))
        self.add_module("layer8_softmax", nn.Softmax())
        
    def forward(self, X):
        return super().forward(X).squeeze()
