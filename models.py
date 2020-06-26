import torch.nn as nn
import torch

class conv_bn(nn.Module):
    def __init__(self, i_f, o_f, fs):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(i_f, o_f, fs, stride = (2, 1))
        self.act = nn.ELU()
        self.bn = nn.BatchNorm2d(o_f)
        #self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride= (2, 1))
    def forward(self, x):
        x = self.bn(self.act(self.conv(x)))
        #return self.pool(x)
        return x
    
    
class conv_block(nn.Module):
    def __init__(self, h_list, input_shape, fs):
        '''
        input_shape : not include batch_size
        '''
        
        super(conv_block, self).__init__()
        self.input_shape = input_shape
        self.fs = fs
        convs = []
        for i in range(len(h_list)):
            if i == 0:
                convs.append(conv_bn(self.input_shape[0], h_list[i], fs))
            else:
                convs.append(conv_bn(h_list[i-1], h_list[i], fs))
        self.convs = nn.Sequential(*convs)
    
    def forward(self, x):
        return self.convs(x)
    

class classifier(nn.Module):
    def __init__(self, h_list, input_size, output_size):
        super(classifier, self).__init__()
        layers = []
        for i in range(len(h_list)):
            if i == 0:
                layers.append(nn.Linear(input_size, h_list[0]))
            else:
                layers.append(nn.Linear(h_list[i-1], h_list[i]))
            layers.append(nn.ELU())
            
        layers.append(nn.Linear(h_list[i], output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class cnn_model(nn.Module):
    def __init__(self, cnn_block, fc_block):
        super(cnn_model, self).__init__()
        self.cnn = cnn_block
        self.fc = fc_block
    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(start_dim = 1)
        return self.fc(x)
    
class cnn_parallel(nn.Module):
    def __init__(self, cnn_blocks, fc_block):
        super(cnn_parallel, self).__init__()
        self.blocks = cnn_blocks
        self.fc = fc_block
        
    def forward(self, x):
        #x_p = [b(x).flatten(start_dim = 1) for b in self.blocks]
        x_p = []
        for b in self.blocks:
            x_p.append(b(x).flatten(start_dim = 1))
        x_p = torch.cat(x_p, axis = -1)
        return self.fc(x_p)