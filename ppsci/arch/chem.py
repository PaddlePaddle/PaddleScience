from paddle import nn
import paddle
from ppsci.arch import base

class ChemMultimodalMLP(base.Arch):
	def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim):
		super(ChemMultimodalMLP, self).__init__()
		
		# 图像模态处理
		self.r1_fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			# nn.Dropout(p=0.4),
			nn.Linear(hidden_dim, hidden_dim2),
			nn.ReLU(),
			nn.Linear(hidden_dim2, hidden_dim3),
		)
		
		# 文本模态处理
		self.r2_fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			# nn.Dropout(p=0.4),
			nn.Linear(hidden_dim, hidden_dim2),
			nn.ReLU(),
			nn.Linear(hidden_dim2, hidden_dim3),
		)
		
		self.ligand_fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
		                               nn.ReLU(),
		                               nn.Linear(hidden_dim, hidden_dim2),
		                               # nn.Dropout(p=0.4),
		                               nn.ReLU(),
		                               nn.Linear(hidden_dim2, hidden_dim3),
		                               )
		
		self.base_fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
		                             nn.ReLU(),
		                             nn.Linear(hidden_dim, hidden_dim2),
		                             nn.ReLU(),
		                             nn.Linear(hidden_dim2, hidden_dim3),
		                             )
		
		self.solvent_fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
		                                nn.ReLU(),
		                                # nn.Dropout(p=0.4),
		                                nn.Linear(hidden_dim, hidden_dim2),
		                                nn.ReLU(),
		                                nn.Linear(hidden_dim2, hidden_dim3),
		                                nn.ReLU(),
		                                )
		
		self.weights = paddle.create_parameter(
			shape=[5],
			dtype='float32',
			default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
		)
		
		# 结合两个模态的输出7
		self.fc_combined = nn.Sequential(
			nn.Linear(hidden_dim3, hidden_dim4),
			nn.ReLU(),
			nn.Linear(hidden_dim4, output_dim),
		)
	
	def weighted_average(self, features, weights):
		"""
		计算加权平均。

		Args:
			features (list of torch.Tensor): 每个模态的特征张量。
			weights (list of float): 每个模态的权重。

		Returns:
			torch.Tensor: 加权平均后的特征张量。
		"""
		# 确保权重是 torch.Tensor 并且与特征的维度一致
		weights = weights.clone().detach()
		
		# 计算加权和
		weighted_sum = sum(f * w for f, w in zip(features, weights))
		
		# 计算权重和
		total_weight = weights.sum()
		
		# 返回加权平均
		return weighted_sum / total_weight
	
	def forward(self, x):
		x = self.concat_to_tensor(x, ("v"), axis=-1)
  		# 沿列维度（axis=1）均分
		input_splits = paddle.split(x, num_or_sections=5, axis=1)
  
		# 解包为 5 个变量
		r1_input, r2_input, ligand_input, base_input, solvent_input = input_splits
  
		# 处理图像输入
		r1_features = self.r1_fc(r1_input)
		
		# 处理文本输入
		r2_features = self.r2_fc(r2_input)
		
		ligand_features = self.ligand_fc(ligand_input)
		
		base_features = self.base_fc(base_input)
		
		solvent_features = self.solvent_fc(solvent_input)
		
		# 结合特征
		features = [r1_features, r2_features, ligand_features, base_features, solvent_features]
		# combined_features = torch.cat((r1_features, r2_features, ligand_features, base_features, solvent_features), dim=1)
		combined_features = self.weighted_average(features, self.weights)
		# print(combined_features.shape)  # 打印 combined_features 的形状
		
		# 最终预测
		output = self.fc_combined(combined_features)
		output = self.split_to_dict(output, ("u"), axis=-1)
		return output