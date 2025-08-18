import os
import sys
import numpy as np
import pandas as pd
import random
import paddle
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import ppsci


# 随机种子设置，确保与 torch 版对齐
random.seed(0)
np.random.seed(0)
paddle.seed(0)

# 数据路径
script_dir = os.path.dirname(os.path.abspath(__file__))
ulbs_path = os.path.join(script_dir, './xrd_data/ulbs.csv')
lbs_path = os.path.join(script_dir, './xrd_data/lbs.csv')

# 数据增强函数（从paddle_only复制）
def normdata(data):  
    min_x = min(data)
    max_x = max(data)
    norm = max_x - min_x
    data = (data - min_x)/norm
    return data

def data_zero(data):  
    num = len(data)
    for i in range(num):
        if(data[i]<0.1):
            data[i]=0
    return data

def weak_augdata(data):
    w_noise_ratio = 0.1
    w_noise_peak = 0.05
    w_move_gap = 100
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data==0)[0]
        idx_num = len(index)
        noise_num = int(idx_num*w_noise_ratio*np.random.random())
        np.random.shuffle(index)       
        for i in index[:noise_num]: 
            data[i] = np.random.random()*w_noise_peak

    ratio = np.random.random()
    if ratio <= 0.5:
        cut = np.random.randint(50,max(w_move_gap,51),1)[0]
        if ratio <= 0.5:
            out = 4501 - cut
            data = np.append(np.zeros(cut),data[:out])
        else:
            data = np.append(data[cut:],np.zeros(cut))
            
    return data

def strong_augdata(data):
    s_noise_ratio = 0.0
    s_noise_peak = 0.12
    s_move_gap = 500
    s_scaling_ratio = 0.15
    s_elimin_ratio  = 0.15
    ratio = np.random.random()
    scaling_num = 0
    if ratio <=0.5:
        index = np.nonzero(data)[0]
        idx_num = len(index)
        scaling_num = int(idx_num*s_scaling_ratio*np.random.random() )  
        np.random.shuffle(index)       
        for i in index[:scaling_num]: 
            data[i] = np.random.random()*2*data[i]+data[i]

    ratio = np.random.random()
    if ratio <=0.5:
        index = np.nonzero(data)[0]
        idx_num = len(index)
        elimin_num = int(idx_num*s_elimin_ratio*np.random.random() )  
        np.random.shuffle(index)       
        for i in index[:elimin_num]: 
            data[i] = 0

    ratio = np.random.random()
    if ratio <= 0.5:
        ndata =data_zero(data)
        index = np.nonzero(ndata)[0]
        idx_num = len(index)
        old_idx = 0
        gap_left = []
        gap_right = []
        cut = np.random.randint(1,s_move_gap,1)[0]
               
        for i in range(idx_num):
            value = index[i] - old_idx
            if value > cut:
                gap_left.append(old_idx)
                gap_right.append(index[i])                   
            old_idx = index[i]
        
        ratio = np.random.random()
        if ratio <= 0.5:
            if (len(gap_right)!=0):
                np.random.shuffle(gap_right)
                sele_site = gap_right[0]
                out = sele_site - cut
                data = np.concatenate((data[:out],data[sele_site:],np.zeros([cut])),axis=0)
        else:
            if (len(gap_left)!=0):            
                np.random.shuffle(gap_left)
                sele_site = gap_left[0]+1 
                out = sele_site + cut       
                data = np.concatenate((np.zeros([cut]),data[:sele_site],data[out:]),axis=0)
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data==0)[0]
        idx_num = len(index)
        noise_num = int(idx_num*s_noise_ratio*np.random.random() )  
        np.random.shuffle(index)        
        for i in index[:noise_num]: 
            data[i] = np.random.random()*s_noise_peak
           
    return data

def main_strong(dataset):
    dataset = normdata(dataset)    
    dataset = data_zero(dataset)
    data = strong_augdata(dataset)
    dataset = normdata(data)    
    dataset = np.reshape(dataset,(1,len(dataset)))    
    dataset = dataset.astype(np.float32)    
    dataset=paddle.to_tensor(dataset)
    return dataset

def main_weak(dataset):
    dataset = normdata(dataset)
    dataset = data_zero(dataset)  
    data = weak_augdata(dataset)
    dataset = normdata(data) 
    dataset = np.reshape(dataset,(1,len(dataset)))        
    dataset = dataset.astype(np.float32)       
    dataset=paddle.to_tensor(dataset)
    return dataset

def main_eval(data):
    dataset = normdata(data)
    dataset = data_zero(dataset)    
    dataset = np.reshape(dataset,(1,len(dataset)))        
    dataset = dataset.astype(np.float32)       
    dataset=paddle.to_tensor(dataset)
    return dataset

# PPSci风格的数据集类
class XRDDataset(paddle.io.Dataset):
    def __init__(self, data, target, transform=None, is_ulb=False, strong_transform=None):
        super().__init__()
        self.data = data
        self.target = target
        self.transform = transform
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        
        if self.is_ulb:
            # 无标签数据：返回弱增强和强增强两个版本
            x_ulb_w = self.transform(data)
            x_ulb_s = self.strong_transform(data) if self.strong_transform else x_ulb_w
            
            return {
                'idx_ulb': index,
                'x_ulb_w': x_ulb_w,
                'x_ulb_s': x_ulb_s
            }
        else:
            # 有标签数据
            x_lb = self.transform(data)
            y_lb = target
            
            return {
                'idx_lb': index,
                'x_lb': x_lb,
                'y_lb': y_lb
            }

    def __len__(self):
        return len(self.data)

# === FlexMatch 损失函数迁移自 paddle_only ===
class FlexMatchLoss:
    def __init__(self, config):
        self.T = getattr(config, 'T', 0.5)
        self.p_cutoff = getattr(config, 'p_cutoff', 0.95)
        self.hard_label = getattr(config, 'hard_label', True)
        self.thresh_warmup = getattr(config, 'thresh_warmup', True)
        self.lambda_u = getattr(config, 'ulb_loss_ratio', 1.0)
        self.num_classes = getattr(config, 'num_classes', 2)
        self.mask_acc = np.zeros(self.num_classes, dtype=np.float32)
        self.mask_cnt = np.zeros(self.num_classes, dtype=np.float32)
        self.criterion = paddle.nn.CrossEntropyLoss()

    def gen_pseudo_label(self, logits):
        probs = paddle.nn.functional.softmax(logits / self.T, axis=-1)
        if self.hard_label:
            pseudo_label = paddle.argmax(probs, axis=-1)
        else:
            pseudo_label = probs
        max_probs = paddle.max(probs, axis=-1)
        return pseudo_label, max_probs

    def get_mask(self, max_probs, pseudo_label):
        mask = (max_probs >= self.p_cutoff).astype('float32')
        if self.thresh_warmup:
            for c in range(self.num_classes):
                class_mask = (pseudo_label == c).astype('float32')
                self.mask_acc[c] += float((mask * class_mask).sum().numpy())
                self.mask_cnt[c] += float(class_mask.sum().numpy())
        return mask

    def __call__(self, model_output, batch):
        # 有标签数据损失
        if 'x_lb' in batch and 'y_lb' in batch:
            logits_lb = model_output['logits']
            loss_lb = self.criterion(logits_lb, batch['y_lb'])
        else:
            loss_lb = paddle.to_tensor(0.0)

        # 无标签数据损失（FlexMatch 机制）
        if 'x_ulb_w' in batch and 'x_ulb_s' in batch:
            with paddle.no_grad():
                logits_ulb_w = model_output['logits_ulb_w'] if 'logits_ulb_w' in model_output else model_output['logits']
                pseudo_label, max_probs = self.gen_pseudo_label(logits_ulb_w)
                mask = self.get_mask(max_probs, pseudo_label if self.hard_label else paddle.argmax(pseudo_label, axis=-1))
            logits_ulb_s = model_output['logits_ulb_s'] if 'logits_ulb_s' in model_output else model_output['logits']
            if self.hard_label:
                loss_ulb = paddle.nn.functional.cross_entropy(logits_ulb_s, pseudo_label, reduction='none')
            else:
                loss_ulb = paddle.nn.functional.kl_div(
                    paddle.nn.functional.log_softmax(logits_ulb_s, axis=-1),
                    pseudo_label, reduction='none').sum(axis=-1)
            loss_ulb = (loss_ulb * mask).mean() if mask.sum() > 0 else paddle.to_tensor(0.0)
        else:
            loss_ulb = paddle.to_tensor(0.0)
        total_loss = loss_lb + self.lambda_u * loss_ulb
        return {
            'loss': total_loss,
            'loss_lb': loss_lb,
            'loss_ulb': loss_ulb
        }

# 与 torch 版保持一致的文件输出
f = open('diver.txt','w')
file_pre = open('pred.txt','w')

def log_and_print(msg, log_file):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def log_info(message, log_file=None):
    """与 torch 版一致的日志格式"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    msg = f"[{timestamp} INFO] {message}"
    print(msg)
    if log_file is not None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

# 自定义训练器（PPSci风格）
class SemiSupervisedTrainer:
    def __init__(self, config, model, optimizer, loss_fn, save_dir='./saved_models_ppsci'):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 日志文件
        self.log_file = os.path.join(save_dir, 'log.txt')
        
    def train_epoch(self, train_lb_loader, train_ulb_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_loss_lb = 0.0
        total_loss_ulb = 0.0
        num_batches = 0
        
        # 与paddle_only一致的训练循环，tqdm显示每个epoch内部的batch进度
        for batch_idx, (data_lb, data_ulb) in enumerate(tqdm(zip(train_lb_loader, train_ulb_loader), 
                                                           total=min(len(train_lb_loader), len(train_ulb_loader)), 
                                                           desc=f"Epoch {epoch} Iter")):
            # 合并批次数据
            batch = {}
            if data_lb:
                batch.update(data_lb)
            if data_ulb:
                batch.update(data_ulb)
            
            # 前向传播
            model_output = {}
            # 有标签数据
            if 'x_lb' in batch:
                model_output['logits'] = self.model(batch['x_lb'])['logits']
            # 无标签 weak/strong
            if 'x_ulb_w' in batch:
                with paddle.no_grad():
                    model_output['logits_ulb_w'] = self.model(batch['x_ulb_w'])['logits']
                model_output['logits_ulb_s'] = self.model(batch['x_ulb_s'])['logits']
            loss_dict = self.loss_fn(model_output, batch)
            
            # 反向传播
            self.optimizer.clear_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            
            total_loss += float(loss_dict['loss'].numpy())
            total_loss_lb += float(loss_dict['loss_lb'].numpy())
            total_loss_ulb += float(loss_dict['loss_ulb'].numpy())
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'loss_lb': total_loss_lb / num_batches,
            'loss_ulb': total_loss_ulb / num_batches
        }
    
    def evaluate(self, eval_loader, log_file=None):
        self.model.eval()
        y_true = []
        y_pred = []
        
        with paddle.no_grad():
            for batch in eval_loader:
                x = batch['x_lb']
                y = batch['y_lb']
                
                logits = self.model(x)['logits']
                pred = paddle.argmax(logits, axis=1)
                
                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred.numpy().tolist())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 检查数据是否为空
        if len(y_true) == 0 or len(y_pred) == 0:
            log_info("Warning: Empty evaluation data", log_file)
            result_dict = {'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            log_info("confusion matrix", log_file)
            log_info("[]", log_file)
            log_info("evaluation metric", log_file)
            for key, item in result_dict.items():
                log_info(f"{key}: {item:.4f}", log_file)
            self.model.train()
            return result_dict
        
        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        
        # 与 torch 版一致的条件判断和输出
        if cf_mat.size > 0 and cf_mat.shape[0] > 0 and cf_mat.shape[1] > 0:
            if cf_mat[0,0] > 0.6 and cf_mat[1,1] > 0.6:
                print((cf_mat[0,0]+cf_mat[1,1])/2, file=f)
                print(cf_mat, file=f)
        
        # 与paddle_only一致的日志输出
        log_info("confusion matrix", log_file)
        log_info(str(cf_mat), log_file)
        result_dict = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
        log_info("evaluation metric", log_file)
        for key, item in result_dict.items():
            log_info(f"{key}: {item:.4f}", log_file)
        
        self.model.train()
        
        return result_dict
    
    def save_model(self, epoch, f1_score):
        if f1_score > self.best_f1:
            self.best_f1 = f1_score
            self.best_epoch = epoch
            save_path = os.path.join(self.save_dir, f'model_best_epoch_{epoch}.pdparams')
            paddle.save(self.model.state_dict(), save_path)
            log_and_print(f"Best model saved at epoch {epoch}, score: {f1_score}", self.log_file)
    
    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

def split_ssl_data(data, target, lb_num_labels, num_classes, ulb_num_labels=None, include_lb_to_ulb=True):
    # 类别均衡采样，支持 ulb_num_labels 限制
    lb_idx = []
    ulb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        lb_count = lb_num_labels // num_classes
        lb_idx.extend(idx[:lb_count])
        if ulb_num_labels is not None:
            ulb_count = ulb_num_labels // num_classes
            ulb_idx.extend(idx[lb_count:lb_count+ulb_count])
        else:
            ulb_idx.extend(idx[lb_count:])
    lb_idx = np.array(lb_idx)
    ulb_idx = np.array(ulb_idx)
    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    lb_data = data[lb_idx]
    lb_target = target[lb_idx]
    ulb_data = data[ulb_idx]
    ulb_target = target[ulb_idx]
    return lb_data, lb_target, ulb_data, ulb_target

def main():
    print("Starting main function with PPSci framework...")
    
    # 读取数据
    print("Reading data...")
    ulb_dataset = pd.read_csv(ulbs_path)
    print("Unlabeled data loaded")
    img_list_train = np.array(ulb_dataset)
    unlb_data = img_list_train[:,3:]

    lb_dataset = pd.read_csv(lbs_path)  
    print("Labeled data loaded")
    img_list = np.array(lb_dataset)
    np.random.seed(0)
    np.random.shuffle(img_list)
    lb_data = img_list[:,5:] 
    lb_target = img_list[:,4].astype(np.int64)
    lb_name = img_list[:,0]
    lb_id = img_list[:,1]
    
    print("Data preprocessing...")
    # 数据预处理（与paddle_only一致）
    a = 0
    c = 0
    pred_data = []
    pred_target = []
    posi_data = []
    posi_target = []
    nega_data = []
    nega_target = []    
    posi_name = []
    posi_id = []  
    nega_name = []
    nega_id = []      
         
    for i in range(len(lb_target)):
        if(lb_target[i]==0):
            a =a+1
            if(a<20):
                posi_data.append(lb_data[i])
                posi_target.append(lb_target[i])
                posi_name.append(lb_name[i])
                posi_id.append(lb_id[i])            
            else:
                pred_data.append(lb_data[i])
                pred_target.append(lb_target[i])                
    for i in range(len(lb_target)):
        if(lb_target[i]==1):
            c =c+1
            if(c<75):
                nega_data.append(lb_data[i])
                nega_target.append(int(lb_target[i]))
                nega_name.append(lb_name[i])
                nega_id.append(lb_id[i])                  
            else:
                pred_data.append(lb_data[i])
                pred_target.append(lb_target[i])  
    
    un_ratio = 0.8
    print("Starting experiments...")
    
    # 100次实验循环（与paddle_only一致）
    for k in range(100):
        print(f"Starting experiment {k+1}/100")
        
        # 配置参数（与paddle_only一致）
        config_params = {
            'epoch': 100,
            'num_train_iter': 1000,
            'num_eval_iter': 10,
            'lr': 3e-4,
            'batch_size': 32,
            'eval_batch_size': 32,
            'num_labels': 20,
            'num_classes': 2,
            'gpu': 0,
            'save_dir': f'./saved_models_ppsci/exp_{k}',
            'save_name': f'./flexmatch_ppsci/{k}'
        }

        lb_num = int(config_params['num_labels']/2)
        np.random.seed(k)            
        np.random.shuffle(posi_data)
        np.random.shuffle(nega_data)
        np.random.shuffle(posi_target)
        np.random.shuffle(nega_target)    
        np.random.shuffle(unlb_data)

        data = unlb_data[:int(len(unlb_data)*un_ratio)]
        target = np.random.randint(0, 2, int(len(unlb_data)*un_ratio))
        train_data = np.append(posi_data[:lb_num],nega_data[:lb_num]).reshape(lb_num*2,len(lb_data[0]))
        train_target = np.append(posi_target[:lb_num],nega_target[:lb_num])
        train_target = np.array(train_target).astype(np.int64)
        n = len(train_data)+len(data)
        data = np.append(train_data,data).reshape(n,len(lb_data[0]))
        target = np.append(train_target,target)

        print("unlb_data shape:", unlb_data.shape)
        print("train_data shape:", train_data.shape)
        print("拼接后 data shape:", data.shape)
        print("target分布：", np.sum(target==0), np.sum(target==1))
        print("unlabeled数据量：", len(unlb_data[:int(len(unlb_data)*un_ratio)]))
        print("ulb_num_labels:", 10000)

        # 半监督划分
        lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(
            data, target, config_params['num_labels'], config_params['num_classes'], ulb_num_labels=10000, include_lb_to_ulb=True)
        
        lb_count = [np.sum(lb_target == i) for i in range(config_params['num_classes'])]
        ulb_count = [np.sum(ulb_target == i) for i in range(config_params['num_classes'])]
        print("lb count:", lb_count)
        print("ulb count:", ulb_count)
        
        # 使用PPSci的数据集
        lb_dataset = XRDDataset(lb_data, lb_target, transform=main_weak, is_ulb=False)
        ulb_dataset = XRDDataset(ulb_data, ulb_target, transform=main_weak, is_ulb=True, strong_transform=main_strong)
        # === 补齐无标签数据采样，保证每个 epoch 固定 10 个 batch ===
        class RepeatDataset(paddle.io.Dataset):
            def __init__(self, dataset, total_len):
                self.dataset = dataset
                self.total_len = total_len
            def __getitem__(self, idx):
                return self.dataset[idx % len(self.dataset)]
            def __len__(self):
                return self.total_len
        ulb_num_batches = 10
        ulb_dataset = RepeatDataset(ulb_dataset, ulb_num_batches * int(config_params['batch_size'] * 3))
        
        eval_num = len(posi_data) + len(nega_data) - config_params['num_labels']
        eval_data = np.append(posi_data[lb_num:],nega_data[lb_num:]).reshape(eval_num,len(lb_data[0]))
        eval_target = np.append(posi_target[lb_num:],nega_target[lb_num:])
        eval_target = np.array(eval_target).astype(np.int64)
        eval_dataset = XRDDataset(eval_data, eval_target, transform=main_eval, is_ulb=False)
        pred_dataset = XRDDataset(pred_data, pred_target, transform=main_eval, is_ulb=False)

        # 使用与paddle_only一致的DataLoader创建逻辑
        # 首先添加DistributedSamplerPaddle类
        class DistributedSamplerPaddle:
            def __init__(self, dataset, num_replicas=1, rank=0, num_samples=None, seed=0):
                if not isinstance(num_samples, int) or num_samples <= 0:
                    raise ValueError(f"num_samples should be a positive integer, but got num_samples={num_samples}")
                self.dataset = dataset
                self.num_replicas = num_replicas
                self.rank = rank
                self.epoch = 0
                self.total_size = num_samples
                assert num_samples % num_replicas == 0, f'{num_samples} samples cant be evenly distributed among {num_replicas} devices.'
                self.num_samples = int(num_samples // num_replicas)
                self.seed = seed

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __iter__(self):
                n = len(self.dataset)
                g = np.random.RandomState(self.epoch + self.seed)
                n_repeats = self.total_size // n
                n_remain = self.total_size % n
                indices = []
                for _ in range(n_repeats):
                    perm = np.arange(n)
                    g.shuffle(perm)
                    indices.extend(perm.tolist())
                if n_remain > 0:
                    perm = np.arange(n)
                    g.shuffle(perm)
                    indices.extend(perm[:n_remain].tolist())
                assert len(indices) == self.total_size
                # subsample
                indices = indices[self.rank:self.total_size:self.num_replicas]
                assert len(indices) == self.num_samples
                return iter(indices)

            def __len__(self):
                return self.num_samples

        # 使用与paddle_only一致的DataLoader
        lb_indices = list(DistributedSamplerPaddle(lb_dataset, num_replicas=1, rank=0, num_samples=10*config_params['batch_size'], seed=0))
        lb_subset = paddle.io.Subset(lb_dataset, lb_indices)
        train_lb_loader = paddle.io.DataLoader(
            lb_subset, 
            batch_size=config_params['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        train_ulb_loader = paddle.io.DataLoader(
            ulb_dataset, 
            batch_size=int(config_params['batch_size'] * 3), 
            shuffle=True, 
            drop_last=True, 
            num_workers=0
        )
        eval_loader = paddle.io.DataLoader(
            eval_dataset, 
            batch_size=config_params['eval_batch_size'], 
            shuffle=False, 
            drop_last=True, 
            num_workers=0
        )
        pred_loader = paddle.io.DataLoader(
            pred_dataset, 
            batch_size=config_params['eval_batch_size'], 
            shuffle=False, 
            drop_last=True, 
            num_workers=0
        )

        # 数据量统计输出，仿照 torch 版
        print(f"unlabeled data number: {len(ulb_dataset)}, labeled data number: {len(lb_dataset)}")
        print("Create train and test data loaders")
        print(f"[!] data loader keys: train_lb, train_ulb, eval, pred")
        print(f"train_lb_loader 批次数: {len(train_lb_loader)}")
        print(f"train_ulb_loader 批次数: {len(train_ulb_loader)}")
        print(f"eval_loader 批次数: {len(eval_loader)}")
        print(f"pred_loader 批次数: {len(pred_loader)}")

        # 使用PPSci的模型
        model = ppsci.arch.VGG(in_channel=1, num_classes=config_params['num_classes'])

        # 使用PPSci的优化器
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=config_params['lr'],
            weight_decay=0.01
        )

        # 使用PPSci的损失函数
        loss_fn = FlexMatchLoss(config_params)

        # 使用PPSci的Trainer进行训练
        trainer = SemiSupervisedTrainer(config_params, model, optimizer, loss_fn, config_params['save_dir'])
        
        print(f"Starting training for experiment {k+1}")
        
        # 完整的训练循环（与paddle_only一致）
        best_f1 = 0.0
        best_epoch = 0
        max_epoch = config_params['num_train_iter'] // 10  # 每个 epoch 10 个 iter，与 paddle_only 一致
        
        for epoch in range(max_epoch):
            # 与paddle_only一致的epoch输出
            log_and_print(f"Epoch: {epoch}", trainer.log_file)
            
            # 训练一个epoch
            train_result = trainer.train_epoch(train_lb_loader, train_ulb_loader, epoch)
            
            # 每个epoch都评估，与paddle_only一致
            eval_result = trainer.evaluate(eval_loader, log_file=trainer.log_file)
            
            if eval_result['f1'] > best_f1:
                best_f1 = eval_result['f1']
                best_epoch = epoch
                
                # 预测并输出结果，与paddle_only一致
                # 只有当pred_loader不为空时才评估
                if len(pred_loader) > 0:
                    pred_result = trainer.evaluate(pred_loader)
                    print(1, best_f1, file=file_pre)
                    print(2, pred_result['f1'], file=file_pre)
                else:
                    print(1, best_f1, file=file_pre)
                    print(2, 0.0, file=file_pre)  # pred_loader为空时输出0.0
                
                # 保存最佳模型
                trainer.save_model(epoch, eval_result['f1'])
        
        log_and_print("Best acc {:.4f} at epoch {:d}".format(best_f1, best_epoch), trainer.log_file)
        log_and_print("Training finished.", trainer.log_file)
        print(f"Experiment {k+1} completed")

if __name__ == '__main__':
    main()
