import paddle as torch
import paddle
import numpy as np
import paddle.nn.functional as F
import paddle.nn as nn
import random

from typing import Optional, Dict, Tuple, Union, List
from ppsci.arch import base


def init_parameter_uniform(parameter: paddle.base.framework.EagerParamBase, n: int) -> None:
    nn.init.uniform_(parameter, -1/np.sqrt(n), 1/np.sqrt(n))

# inputs, hidden_units, outputs, d_out, sigma, dp_ratio, first_omega_0, hidden_omega_0, reg
class IFMMLP(base.Arch):
    def __init__(self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        hidden_units: List[int],
        embed_name: str,
        inputs: int,
        outputs: int,
        d_out: int,
        sigma: float,
        dp_ratio: float,
        reg,
        first_omega_0,
        hidden_omega_0,
        ):
        super().__init__()
        # hidden_units = [hyper_paras['hidden_unit1'], hyper_paras['hidden_unit2'], hyper_paras['hidden_unit3']]
        self.input_keys = input_keys
        self.output_keys = output_keys

        # initialization
        if embed_name == 'None':
            my_model = MyDNN(inputs=inputs, hidden_units=hidden_units, dp_ratio=dp_ratio, outputs=outputs, reg=reg)
        elif embed_name == 'LE':
            my_model = LE_DNN(inputs=inputs, hidden_units=hidden_units, d_out=d_out + 1, dp_ratio=dp_ratio, outputs=outputs, reg=reg)
        elif embed_name == 'LSIM':
            my_model = LSIM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=d_out + 1, sigma=sigma, dp_ratio=dp_ratio, outputs=outputs, reg=reg)
        elif embed_name == 'IFM':
            my_model = IFM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=d_out + 1, sigma=sigma, dp_ratio=dp_ratio, first_omega_0=first_omega_0,
                            hidden_omega_0=hidden_omega_0, outputs=outputs, reg=reg)
        elif embed_name == 'GM':
            my_model = GM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=d_out + 1, sigma=sigma + 1, dp_ratio=dp_ratio, outputs=outputs, reg=reg)
        elif embed_name == 'SIM':
            my_model = SIM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=d_out + 1, sigma=sigma + 1, dp_ratio=dp_ratio, outputs=outputs, reg=reg)
        else:
            raise ValueError("Invalid Embedding Name")
        
        self.model = my_model

    def forward(self, x):
        Xs = x[self.input_keys[0]]
        #Xs, masks = Xs.to(args.device), Ys.to(args.device), masks.to(args.device)

        ret = self.model(Xs)
        return {self.output_keys[0] : ret}

class MyDNN(nn.Layer):
    def __init__(self, inputs, hidden_units, outputs, dp_ratio, reg):
        """
        :param inputs: number of inputs
        :param hidden_units: [128, 256, 512]
        :param out_puts: number of outputs
        :param dp_ratio:
        :param reg:
        """
        super(MyDNN, self).__init__()
        # parameters
        self.reg = reg

        # layers
        self.hidden1 = nn.Linear(inputs, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

class LE(nn.Layer):
    def __init__(self, n_tokens: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(paddle.to_tensor(n_tokens, 1, d_out))
        self.bias = nn.Parameter(paddle.to_tensor(n_tokens, d_out))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        d_out = self.weight.shape[-1]
        init_parameter_uniform(self.weight, d_out)
        init_parameter_uniform(self.bias, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features, d_in)
        returns: (n_batch, n_features, d_out)
        """
        x = x.unsqueeze(-1)
        x = (x.unsqueeze(-2)@self.weight[None]).squeeze(-2)
        x = x + self.bias[None]
        return x

class PLE(nn.Layer):
    def __init__(self, n_num_features: int, d_out: int, sigma: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        coefficients = paddle.to_tensor(n_num_features, d_out)
        self.coefficients = nn.Parameter(coefficients)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, 0.0, self.sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2*np.pi*self.coefficients[None]*x[..., None]
        return paddle.concat([torch.cos(x), torch.sin(x)], -1)

class LE_DNN(nn.Layer):
    def __init__(self, inputs, hidden_units, outputs, d_out, dp_ratio, reg):
        super(LE_DNN, self).__init__()
        # parameters
        self.reg = reg
        # layers
        self.hidden1 = nn.Linear(inputs * d_out, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
        self.embedding = LE(inputs, d_out)

    def forward(self, x):

        x = self.embedding(x).view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

class LSIM_DNN(nn.Layer):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, reg):
        super(LSIM_DNN, self).__init__()
        # parameters
        self.reg = reg
        # layers
        self.hidden1 = nn.Linear(inputs, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
        self.embedding = PLE(inputs, d_out, sigma)
        self.linear = nn.Linear(d_out * 2, inputs)

    def forward(self, x):
        x = self.embedding(x).sum(1)
        x = F.relu(self.linear(x))
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

class gaussian_encoding(nn.Layer):   
    def __init__(self, n_num_features: int, d_out: int, sigma: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        self.n_num_features = n_num_features
        self.size = (d_out, n_num_features)
        self.B = torch.randn(self.size) * sigma
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features * 2 * d_out)
        """
        self.B = self.B.to(x.device)
        xp = 2 * np.pi * x @ self.B.T
        return paddle.concat((torch.cos(xp), torch.sin(xp)), axis=-1)
    
class GM_DNN(nn.Layer):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, reg):
        """
        :param inputs: number of inputs
        :param hidden_units: [128, 256, 512]
        :param out_puts: number of outputs
        :param dp_ratio:
        :param reg:
        """
        super(GM_DNN, self).__init__()
        # parameters
        self.reg = reg
        self.d_out = d_out
        self.sigma = sigma
        # layers
        self.hidden1 = nn.Linear(d_out * 2, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)

        self.embedding = gaussian_encoding(inputs, d_out, sigma)


    def forward(self, x):
        x = self.embedding(x)
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)
    
class SineLayer(nn.Layer):    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias_attr=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class IFM_DNN(nn.Layer):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, first_omega_0, hidden_omega_0, reg):
        """
        :param inputs: number of inputs
        :param hidden_units
        :param out_puts: number of outputs
        :param dp_ratio:
        :param reg:
        """
        super(IFM_DNN, self).__init__()
        # parameters
        self.reg = reg
        # layers
        self.hidden1 = SineLayer(inputs, hidden_units[0], is_first=True, omega_0=first_omega_0)
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = SineLayer(hidden_units[0], hidden_units[1], is_first=False, omega_0=hidden_omega_0)
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = SineLayer(hidden_units[1], hidden_units[2], is_first=False, omega_0=hidden_omega_0)
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
            with torch.no_grad():
                self.output.weight.uniform_(-np.sqrt(6 / hidden_units[2]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_units[2]) / hidden_omega_0)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
            with torch.no_grad():
                self.output.weight.uniform_(-np.sqrt(6 / hidden_units[2]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_units[2]) / hidden_omega_0)

    def forward(self, x):

        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))
        #x = self.dropout1(x)

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))
        #x = self.dropout2(x)

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))
        #x = self.dropout3(x)

        return self.output(x)

class SIM_encoding(nn.Layer):
    def __init__(self, n_num_features: int, d_out: int, sigma: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        self.n_num_features = n_num_features
        self.coeffs = 2 * np.pi * sigma ** (torch.arange(d_out) / d_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features * 2 * d_out)
        """
        xp = self.coeffs.to(x.device) * torch.unsqueeze(x, -1)
        xp_cat = paddle.concat((torch.cos(xp), torch.sin(xp)), axis=-1)
        return xp_cat.flatten(-2, -1) 

class SIM_DNN(nn.Layer):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, reg):
        """
        :param inputs: number of inputs
        :param hidden_units
        :param out_puts: number of outputs   m
        :param dp_ratio:
        :param reg:
        """
        super(SIM_DNN, self).__init__()
        # parameters
        self.reg = reg
        self.d_out = d_out
        self.sigma = sigma
        # layers
        self.hidden1 = nn.Linear(d_out * 2 * inputs, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
        self.embedding = SIM_encoding(inputs, d_out, sigma)

    def forward(self, x):
        x = self.embedding(x)
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

def collate_fn(data_batch):
    Xs, Ys, masks = map(list, zip(*data_batch))

    Xs = torch.stack(Xs, axis=0)
    Ys = torch.stack(Ys, axis=0)
    masks = torch.stack(masks, axis=0)

    return Xs, Ys, masks

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子