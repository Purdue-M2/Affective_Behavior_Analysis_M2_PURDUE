import torch.nn as nn

class AuModel(nn.Module):
    def __init__(self):
        super(AuModel, self).__init__()

        dropout_rate = 0.6

        
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

        )

        # 
        self.output_layer = nn.Linear(768, 12)

        #he initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):

        x = self.layers(x)

        output = self.output_layer(x)


        return output
    
class DFADModel(nn.Module):
    def __init__(self):
        super(DFADModel, self).__init__()

        dropout_rate = 0.6
        leaky_relu_slope = 0.01 

        
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

        )

        # 
        self.output_layer = nn.Linear(768, 8)

        #he initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):

        x = self.layers(x)

        output = self.output_layer(x)


        return output
    





if __name__ == '__main__':

    model = DFADModel()  
    print(model)  
