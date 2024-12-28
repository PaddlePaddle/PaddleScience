import os
import paddle
from __future__ import annotations
import datetime
import inspect
import random
import shutil
import time
from typing import TYPE_CHECKING, Literal, get_args
import numpy as np
from chgnet.model.model import CHGNet
from chgnet.utils import AverageMeter, determine_device, mae, write_json
try:
    import wandb
except ImportError:
    wandb = None
if TYPE_CHECKING:
    pass
    from typing_extensions import Self
    from chgnet import TrainTask
LogFreq = Literal['epoch', 'batch']
LogEachEpoch, LogEachBatch = get_args(LogFreq)


class Trainer:
    """A trainer to train CHGNet using energy, force, stress and magmom."""

    def __init__(self, model: (CHGNet | None)=None, *, targets: TrainTask=
        'ef', energy_loss_ratio: float=1, force_loss_ratio: float=1,
        stress_loss_ratio: float=0.1, mag_loss_ratio: float=0.1, optimizer:
        str='Adam', scheduler: str='CosLR', criterion: str='MSE', epochs:
        int=50, starting_epoch: int=0, learning_rate: float=0.001,
        print_freq: int=100, torch_seed: (int | None)=None, data_seed: (int |
        None)=None, use_device: (str | None)=None, check_cuda_mem: bool=
        False, wandb_path: (str | None)=None, wandb_init_kwargs: (dict |
        None)=None, extra_run_config: (dict | None)=None, **kwargs) ->None:
        """Initialize all hyper-parameters for trainer.

        Args:
            model (nn.Module): a CHGNet model
            targets ("ef" | "efs" | "efsm"): The training targets. Default = "ef"
            energy_loss_ratio (float): energy loss ratio in loss function
                Default = 1
            force_loss_ratio (float): force loss ratio in loss function
                Default = 1
            stress_loss_ratio (float): stress loss ratio in loss function
                Default = 0.1
            mag_loss_ratio (float): magmom loss ratio in loss function
                Default = 0.1
            optimizer (str): optimizer to update model. Can be "Adam", "SGD", "AdamW",
                "RAdam". Default = 'Adam'
            scheduler (str): learning rate scheduler. Can be "CosLR", "ExponentialLR",
                "CosRestartLR". Default = 'CosLR'
            criterion (str): loss function criterion. Can be "MSE", "Huber", "MAE"
                Default = 'MSE'
            epochs (int): number of epochs for training
                Default = 50
            starting_epoch (int): The epoch number to start training at.
            learning_rate (float): initial learning rate
                Default = 1e-3
            print_freq (int): frequency to print training output
                Default = 100
            torch_seed (int): random seed for torch
                Default = None
            data_seed (int): random seed for random
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            wandb_path (str | None): The project and run name separated by a slash:
                "project/run_name". If None, wandb logging is not used.
                Default = None
            wandb_init_kwargs (dict): Additional kwargs to pass to wandb.init.
                Default = None
            extra_run_config (dict): Additional hyper-params to be recorded by wandb
                that are not included in the trainer_args. Default = None

            **kwargs (dict): additional hyper-params for optimizer, scheduler, etc.

        Raises:
            NotImplementedError: If the optimizer or scheduler is not implemented
            ImportError: If wandb_path is specified but wandb is not installed
            ValueError: If wandb_path is specified but not in the format
                'project/run_name'
        """
        self.trainer_args = {k: v for k, v in locals().items() if k not in
            {'self', '__class__', 'model', 'kwargs'}} | kwargs
        self.model = model
        self.targets = targets
        if torch_seed is not None:
            paddle.seed(seed=torch_seed)
        if data_seed:
            random.seed(data_seed)
        if optimizer == 'SGD':
            momentum = kwargs.pop('momentum', 0.9)
            weight_decay = kwargs.pop('weight_decay', 0)
            self.optimizer = paddle.optimizer.SGD(parameters=model.parameters(),
                learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            weight_decay = kwargs.pop('weight_decay', 0)
            self.optimizer = paddle.optimizer.Adam(parameters=model.
                parameters(), learning_rate=learning_rate, weight_decay=
                weight_decay)
        elif optimizer == 'AdamW':
            weight_decay = kwargs.pop('weight_decay', 0.01)
            self.optimizer = paddle.optimizer.AdamW(parameters=model.
                parameters(), learning_rate=learning_rate, weight_decay=
                weight_decay)
        elif optimizer == 'RAdam':
            weight_decay = kwargs.pop('weight_decay', 0)
            self.optimizer = paddle.optimizer.RAdam(model.parameters(),
                learning_rate, weight_decay=weight_decay)
        default_decay_frac = 0.01
        if scheduler in {'MultiStepLR', 'multistep'}:
            scheduler_params = kwargs.pop('scheduler_params', {'milestones':
                [4 * epochs, 6 * epochs, 8 * epochs, 9 * epochs], 'gamma': 0.3}
                )
            self.scheduler = paddle.optimizer.lr.MultiStepDecay(self.
                optimizer, **scheduler_params)
            self.scheduler_type = 'multistep'
        elif scheduler in {'ExponentialLR', 'Exp', 'Exponential'}:
            scheduler_params = kwargs.pop('scheduler_params', {'gamma': 0.98})
            self.scheduler = paddle.optimizer.lr.ExponentialDecay(self.
                optimizer, **scheduler_params)
            self.scheduler_type = 'exp'
        elif scheduler in {'CosineAnnealingLR', 'CosLR', 'Cos', 'cos'}:
            scheduler_params = kwargs.pop('scheduler_params', {
                'decay_fraction': default_decay_frac})
            decay_fraction = scheduler_params.pop('decay_fraction',
                default_decay_frac)
            tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=10 *
                epochs, eta_min=decay_fraction * learning_rate,
                learning_rate=self.optimizer.get_lr())
            self.optimizer.set_lr_scheduler(tmp_lr)
            self.scheduler = tmp_lr
            self.scheduler_type = 'cos'
        elif scheduler == 'CosRestartLR':
            scheduler_params = kwargs.pop('scheduler_params', {
                'decay_fraction': default_decay_frac, 'T_0': 10, 'T_mult': 2})
            decay_fraction = scheduler_params.pop('decay_fraction',
                default_decay_frac)
            self.scheduler = (torch.optim.lr_scheduler.
                CosineAnnealingWarmRestarts(self.optimizer, eta_min=
                decay_fraction * learning_rate, **scheduler_params))
            self.scheduler_type = 'cosrestart'
        else:
            raise NotImplementedError
        self.criterion = CombinedLoss(target_str=self.targets, criterion=
            criterion, is_intensive=self.model.is_intensive,
            energy_loss_ratio=energy_loss_ratio, force_loss_ratio=
            force_loss_ratio, stress_loss_ratio=stress_loss_ratio,
            mag_loss_ratio=mag_loss_ratio, **kwargs)
        self.epochs = epochs
        self.starting_epoch = starting_epoch
        self.device = determine_device(use_device=use_device,
            check_cuda_mem=check_cuda_mem)
        self.print_freq = print_freq
        self.training_history: dict[str, dict[Literal['train', 'val',
            'test'], list[float]]] = {key: {'train': [], 'val': [], 'test':
            []} for key in self.targets}
        self.best_model = None
        if wandb_path:
            if wandb is None:
                raise ImportError(
                    'Weights and Biases not installed. pip install wandb to use wandb logging.'
                    )
            if wandb_path.count('/') == 1:
                project, run_name = wandb_path.split('/')
            else:
                raise ValueError(
                    f"wandb_path={wandb_path!r} should be in the format 'project/run_name' (no extra slashes)"
                    )
            wandb.init(project=project, name=run_name, config=self.
                trainer_args | (extra_run_config or {}), **
                wandb_init_kwargs or {})

    def train(self, train_loader: paddle.io.DataLoader, val_loader: paddle.
        io.DataLoader, test_loader: (DataLoader | None)=None, *, save_dir:
        (str | None)=None, save_test_result: bool=False,
        train_composition_model: bool=False, wandb_log_freq: LogFreq=
        LogEachBatch) ->None:
        """Train the model using torch data_loaders.

        Args:
            train_loader (DataLoader): train loader to update CHGNet weights
            val_loader (DataLoader): val loader to test accuracy after each epoch
            test_loader (DataLoader):  test loader to test accuracy at end of training.
                Can be None.
                Default = None
            save_dir (str): the dir name to save the trained weights
                Default = None
            save_test_result (bool): Whether to save the test set prediction in a JSON
                file. Default = False
            train_composition_model (bool): whether to train the composition model
                (AtomRef), this is suggested when the fine-tuning dataset has large
                elemental energy shift from the pretrained CHGNet, which typically comes
                from different DFT pseudo-potentials.
                Default = False
            wandb_log_freq ("epoch" | "batch"): Frequency of logging to wandb.
                'epoch' logs once per epoch, 'batch' logs after every batch.
                Default = "batch"

        Raises:
            ValueError: If model is not initialized
        """
        if self.model is None:
            raise ValueError('Model needs to be initialized')
        global best_checkpoint
        if save_dir is None:
            save_dir = (
                f'{datetime.datetime.now(tz=datetime.timezone.utc):%m-%d-%Y}')
        print(f'Begin Training: using {self.device} device')
        print(f'training targets: {self.targets}')
        self.model.to(self.device)
        for param in self.model.composition_model.parameters():
            param.stop_gradient = not train_composition_model
        for epoch in range(self.starting_epoch, self.epochs):
            train_mae = self._train(train_loader, epoch, wandb_log_freq)
            if 'e' in train_mae and train_mae['e'] != train_mae['e']:
                print('Exit due to NaN')
                break
            val_mae = self._validate(val_loader, is_test=False,
                wandb_log_freq=wandb_log_freq)
            for key in self.targets:
                self.training_history[key]['train'].append(train_mae[key])
                self.training_history[key]['val'].append(val_mae[key])
            if 'e' in val_mae and val_mae['e'] != val_mae['e']:
                print('Exit due to NaN')
                break
            if save_dir:
                self.save_checkpoint(epoch, val_mae, save_dir=save_dir)
            if (wandb is not None and wandb_log_freq == LogEachEpoch and
                self.trainer_args.get('wandb_path')):
                wandb.log({f'train_{k}_mae': v for k, v in train_mae.items(
                    )} | {f'val_{k}_mae': v for k, v in val_mae.items()} |
                    {'epoch': epoch})
        if test_loader is not None:
            print('---------Evaluate Model on Test Set---------------')
            for file in os.listdir(save_dir):
                if file.startswith('bestE_'):
                    test_file = file
                    best_checkpoint = paddle.load(path=str(os.path.join(
                        save_dir, test_file)))
            self.model.set_state_dict(state_dict=best_checkpoint['model'][
                'state_dict'])
            test_mae = self._validate(test_loader, is_test=True,
                test_result_save_path=save_dir if save_test_result else None)
            for key in self.targets:
                self.training_history[key]['test'] = test_mae[key]
            self.save(filename=os.path.join(save_dir, test_file))
            if wandb is not None and self.trainer_args.get('wandb_path'):
                wandb.log({f'test_{k}_mae': v for k, v in test_mae.items()})

    def _train(self, train_loader: paddle.io.DataLoader, current_epoch: int,
        wandb_log_freq: LogFreq=LogEachBatch) ->dict:
        """Train all data for one epoch.

        Args:
            train_loader (DataLoader): train loader to update CHGNet weights
            current_epoch (int): used for resume unfinished training
            wandb_log_freq ("epoch" | "batch"): Frequency of logging to wandb

        Returns:
            dictionary of training errors
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = {}
        for target in self.targets:
            mae_errors[target] = AverageMeter()
        self.model.train()
        start = time.perf_counter()
        for idx, (graphs, targets) in enumerate(train_loader):
            data_time.update(time.perf_counter() - start)
            for g in graphs:
                requires_force = 'f' in self.targets
                g.atom_frac_coord.stop_gradient = not requires_force
            graphs = [g.to(self.device) for g in graphs]
            targets = {k: self.move_to(v, self.device) for k, v in targets.
                items()}
            prediction = self.model(graphs, task=self.targets)
            combined_loss = self.criterion(targets, prediction)
            losses.update(combined_loss['loss'].data.cpu().item(), len(graphs))
            for key in self.targets:
                mae_errors[key].update(combined_loss[f'{key}_MAE'].cpu().
                    item(), combined_loss[f'{key}_MAE_size'])
            self.optimizer.clear_gradients(set_to_zero=False)
            combined_loss['loss'].backward()
            self.optimizer.step()
            if idx + 1 in np.arange(1, 11) * len(train_loader) // 10:
                self.scheduler.step()
            del graphs, targets
            del prediction, combined_loss
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()
            if idx == 0 or (idx + 1) % self.print_freq == 0:
                message = (
                    f'Epoch: [{current_epoch}][{idx + 1}/{len(train_loader)}] | Time ({batch_time.avg:.3f})({data_time.avg:.3f}) | Loss {losses.val:.4f}({losses.avg:.4f}) | MAE '
                    )
                for key in self.targets:
                    message += (
                        f'{key} {mae_errors[key].val:.3f}({mae_errors[key].avg:.3f})  '
                        )
                print(message)
            if (wandb is not None and wandb_log_freq == 'batch' and self.
                trainer_args.get('wandb_path')):
                wandb.log({f'train_{k}_mae': v.avg for k, v in mae_errors.
                    items()} | {'train_loss': losses.avg, 'epoch':
                    current_epoch, 'batch': idx})
        return {key: round(err.avg, 6) for key, err in mae_errors.items()}

    def _validate(self, val_loader: paddle.io.DataLoader, *, is_test: bool=
        False, test_result_save_path: (str | None)=None, wandb_log_freq:
        LogFreq=LogEachBatch) ->dict:
        """Validation or test step.

        Args:
            val_loader (DataLoader): val loader to test accuracy after each epoch
            is_test (bool): whether it's test step
            test_result_save_path (str): path to save test_result
            wandb_log_freq ("epoch" | "batch"): Frequency of logging to wandb.
                'epoch' logs once per epoch, 'batch' logs after every batch.

        Returns:
            dictionary of training errors
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = {}
        for key in self.targets:
            mae_errors[key] = AverageMeter()
        self.model.eval()
        if is_test:
            test_pred = []
        end = time.perf_counter()
        for ii, (graphs, targets) in enumerate(val_loader):
            if 'f' in self.targets or 's' in self.targets:
                for graph in graphs:
                    requires_force = 'f' in self.targets
                    graph.atom_frac_coord.stop_gradient = not requires_force
                graphs = [g.to(self.device) for g in graphs]
                targets = {k: self.move_to(v, self.device) for k, v in
                    targets.items()}
            else:
                with paddle.no_grad():
                    graphs = [g.to(self.device) for g in graphs]
                    targets = {k: self.move_to(v, self.device) for k, v in
                        targets.items()}
            prediction = self.model(graphs, task=self.targets)
            combined_loss = self.criterion(targets, prediction)
            losses.update(combined_loss['loss'].data.cpu().item(), len(graphs))
            for key in self.targets:
                mae_errors[key].update(combined_loss[f'{key}_MAE'].cpu().
                    item(), combined_loss[f'{key}_MAE_size'])
            if is_test and test_result_save_path:
                for jj, graph_i in enumerate(graphs):
                    tmp = {'mp_id': graph_i.mp_id, 'graph_id': graph_i.
                        graph_id, 'energy': {'ground_truth': targets['e'][
                        jj].cpu().detach().tolist(), 'prediction':
                        prediction['e'][jj].cpu().detach().tolist()}}
                    if 'f' in self.targets:
                        tmp['force'] = {'ground_truth': targets['f'][jj].
                            cpu().detach().tolist(), 'prediction':
                            prediction['f'][jj].cpu().detach().tolist()}
                    if 's' in self.targets:
                        tmp['stress'] = {'ground_truth': targets['s'][jj].
                            cpu().detach().tolist(), 'prediction':
                            prediction['s'][jj].cpu().detach().tolist()}
                    if 'm' in self.targets:
                        if targets['m'][jj] is not None:
                            m_ground_truth = targets['m'][jj].cpu().detach(
                                ).tolist()
                        else:
                            m_ground_truth = None
                        tmp['mag'] = {'ground_truth': m_ground_truth,
                            'prediction': prediction['m'][jj].cpu().detach(
                            ).tolist()}
                    test_pred.append(tmp)
            del graphs, targets
            del prediction, combined_loss
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()
            if (ii + 1) % self.print_freq == 0:
                name = 'Test' if is_test else 'Val'
                message = (
                    f'{name}: [{ii + 1}/{len(val_loader)}] | Time ({batch_time.avg:.3f}) | Loss {losses.val:.4f}({losses.avg:.4f}) | MAE '
                    )
                for key in self.targets:
                    message += (
                        f'{key} {mae_errors[key].val:.3f}({mae_errors[key].avg:.3f})  '
                        )
                print(message)
            if (wandb is not None and not is_test and wandb_log_freq ==
                'batch' and self.trainer_args.get('wandb_path')):
                wandb.log({f'val_{k}_mae': v.avg for k, v in mae_errors.
                    items()} | {'val_loss': losses.avg, 'batch': ii})
        if is_test:
            message = '**  '
            if test_result_save_path:
                write_json(test_pred, os.path.join(test_result_save_path,
                    'test_result.json'))
        else:
            message = '*   '
        for key in self.targets:
            message += f'{key}_MAE ({mae_errors[key].avg:.3f}) \t'
        print(message)
        if (wandb is not None and not is_test and wandb_log_freq ==
            LogEachEpoch and self.trainer_args.get('wandb_path')):
            wandb.log({f'val_{k}_mae': v.avg for k, v in mae_errors.items()})
        return {k: round(mae_error.avg, 6) for k, mae_error in mae_errors.
            items()}

    def get_best_model(self) ->CHGNet:
        """Get best model recorded in the trainer.

        Returns:
            CHGNet: the model with lowest validation set energy error
        """
        if self.best_model is None:
            raise RuntimeError('the model needs to be trained first')
        MAE = min(self.training_history['e']['val'])
        print(f'Best model has val MAE ={MAE:.4}')
        return self.best_model

    @property
    def _init_keys(self) ->list[str]:
        return [key for key in list(inspect.signature(Trainer.__init__).
            parameters) if key not in {'self', 'model', 'kwargs'}]

    def save(self, filename: str='training_result.pth.tar') ->None:
        """Save the model, graph_converter, etc."""
        state = {'model': self.model.as_dict(), 'optimizer': self.optimizer
            .state_dict(), 'scheduler': self.scheduler.state_dict(),
            'training_history': self.training_history, 'trainer_args': self
            .trainer_args}
        paddle.save(obj=state, path=filename)

    def save_checkpoint(self, epoch: int, mae_error: dict, save_dir: str
        ) ->None:
        """Function to save CHGNet trained weights after each epoch.

        Args:
            epoch (int): the epoch number
            mae_error (dict): dictionary that stores the MAEs
            save_dir (str): the directory to save trained weights
        """
        os.makedirs(save_dir, exist_ok=True)
        for fname in os.listdir(save_dir):
            if fname.startswith('epoch'):
                os.remove(os.path.join(save_dir, fname))
        err_str = '_'.join(
            f"{key}{f'{mae_error[key] * 1000:.0f}' if key in mae_error else 'NA'}"
             for key in 'efsm')
        filename = os.path.join(save_dir, f'epoch{epoch}_{err_str}.pth.tar')
        self.save(filename=filename)
        if mae_error['e'] == min(self.training_history['e']['val']):
            self.best_model = self.model
            for fname in os.listdir(save_dir):
                if fname.startswith('bestE'):
                    os.remove(os.path.join(save_dir, fname))
            shutil.copyfile(filename, os.path.join(save_dir,
                f'bestE_epoch{epoch}_{err_str}.pth.tar'))
        if 'f' in self.targets and mae_error['f'] == min(self.
            training_history['f']['val']):
            for fname in os.listdir(save_dir):
                if fname.startswith('bestF'):
                    os.remove(os.path.join(save_dir, fname))
            shutil.copyfile(filename, os.path.join(save_dir,
                f'bestF_epoch{epoch}_{err_str}.pth.tar'))

    @classmethod
    def load(cls, path: str) ->Self:
        """Load trainer state_dict.

        Args:
            path (str): path to the saved model

        Returns:
            Trainer: the loaded trainer
        """
        state = paddle.load(path=str(path))
        model = CHGNet.from_dict(state['model'])
        print(
            f'Loaded model params = {sum(p.size for p in model.parameters()):,}'
            )
        state['trainer_args'].pop('model', None)
        trainer = cls(model=model, **state['trainer_args'])
        trainer.model.to(trainer.place)
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.scheduler.load_state_dict(state['scheduler'])
        trainer.training_history = state['training_history']
        trainer.starting_epoch = len(trainer.training_history['e']['train'])
        return trainer

    @staticmethod
    def move_to(obj: (Tensor | list[paddle.Tensor]), device: (paddle.
        CPUPlace, paddle.CUDAPlace, str)) ->(Tensor | list[paddle.Tensor]):
        """Move object to device.

        Args:
            obj (Tensor | list[Tensor]): object(s) to move to device
            device (torch.device): device to move object to

        Raises:
            TypeError: if obj is not a tensor or list of tensors

        Returns:
            Tensor | list[Tensor]: moved object(s)
        """
        if paddle.is_tensor(x=obj):
            return obj.to(device)
        if isinstance(obj, list):
            out = []
            for tensor in obj:
                if tensor is not None:
                    out.append(tensor.to(device))
                else:
                    out.append(None)
            return out
        raise TypeError('Invalid type for move_to')


class CombinedLoss(paddle.nn.Layer):
    """A combined loss function of energy, force, stress and magmom."""

    def __init__(self, *, target_str: str='ef', criterion: str='MSE',
        is_intensive: bool=True, energy_loss_ratio: float=1,
        force_loss_ratio: float=1, stress_loss_ratio: float=0.1,
        mag_loss_ratio: float=0.1, delta: float=0.1) ->None:
        """Initialize the combined loss.

        Args:
            target_str: the training target label. Can be "e", "ef", "efs", "efsm" etc.
                Default = "ef"
            criterion: loss criterion to use
                Default = "MSE"
            is_intensive (bool): whether the energy label is intensive
                Default = True
            energy_loss_ratio (float): energy loss ratio in loss function
                Default = 1
            force_loss_ratio (float): force loss ratio in loss function
                Default = 1
            stress_loss_ratio (float): stress loss ratio in loss function
                Default = 0.1
            mag_loss_ratio (float): magmom loss ratio in loss function
                Default = 0.1
            delta (float): delta for torch.nn.HuberLoss. Default = 0.1
        """
        super().__init__()
        if criterion in {'MSE', 'mse'}:
            self.criterion = paddle.nn.MSELoss()
        elif criterion in {'MAE', 'mae', 'l1'}:
            self.criterion = paddle.nn.L1Loss()
        elif criterion == 'Huber':
            self.criterion = paddle.nn.SmoothL1Loss(delta=delta)
        else:
            raise NotImplementedError
        self.target_str = target_str
        self.is_intensive = is_intensive
        self.energy_loss_ratio = energy_loss_ratio
        if 'f' not in self.target_str:
            self.force_loss_ratio = 0
        else:
            self.force_loss_ratio = force_loss_ratio
        if 's' not in self.target_str:
            self.stress_loss_ratio = 0
        else:
            self.stress_loss_ratio = stress_loss_ratio
        if 'm' not in self.target_str:
            self.mag_loss_ratio = 0
        else:
            self.mag_loss_ratio = mag_loss_ratio

    def forward(self, targets: dict[str, paddle.Tensor], prediction: dict[
        str, paddle.Tensor]) ->dict[str, paddle.Tensor]:
        """Compute the combined loss using CHGNet prediction and labels
        this function can automatically mask out magmom loss contribution of
        data points without magmom labels.

        Args:
            targets (dict): DFT labels
            prediction (dict): CHGNet prediction

        Returns:
            dictionary of all the loss, MAE and MAE_size
        """
        out = {'loss': 0.0}
        if 'e' in self.target_str:
            if self.is_intensive:
                out['loss'] += self.energy_loss_ratio * self.criterion(targets
                    ['e'], prediction['e'])
                out['e_MAE'] = mae(targets['e'], prediction['e'])
                out['e_MAE_size'] = tuple(prediction['e'].shape)[0]
            else:
                e_per_atom_target = targets['e'] / prediction['atoms_per_graph'
                    ]
                e_per_atom_pred = prediction['e'] / prediction[
                    'atoms_per_graph']
                out['loss'] += self.energy_loss_ratio * self.criterion(
                    e_per_atom_target, e_per_atom_pred)
                out['e_MAE'] = mae(e_per_atom_target, e_per_atom_pred)
                out['e_MAE_size'] = tuple(prediction['e'].shape)[0]
        if 'f' in self.target_str:
            forces_pred = paddle.concat(x=prediction['f'], axis=0)
            forces_target = paddle.concat(x=targets['f'], axis=0)
            out['loss'] += self.force_loss_ratio * self.criterion(forces_target
                , forces_pred)
            out['f_MAE'] = mae(forces_target, forces_pred)
            out['f_MAE_size'] = tuple(forces_target.shape)[0]
        if 's' in self.target_str:
            stress_pred = paddle.concat(x=prediction['s'], axis=0)
            stress_target = paddle.concat(x=targets['s'], axis=0)
            out['loss'] += self.stress_loss_ratio * self.criterion(
                stress_target, stress_pred)
            out['s_MAE'] = mae(stress_target, stress_pred)
            out['s_MAE_size'] = tuple(stress_target.shape)[0]
        if 'm' in self.target_str:
            mag_preds, mag_targets = [], []
            m_mae_size = 0
            for mag_pred, mag_target in zip(prediction['m'], targets['m'],
                strict=True):
                if mag_target is not None:
                    mag_preds.append(mag_pred)
                    mag_targets.append(mag_target)
                    m_mae_size += tuple(mag_target.shape)[0]
            if mag_targets != []:
                mag_preds = paddle.concat(x=mag_preds, axis=0)
                mag_targets = paddle.concat(x=mag_targets, axis=0)
                out['loss'] += self.mag_loss_ratio * self.criterion(mag_targets
                    , mag_preds)
                out['m_MAE'] = mae(mag_targets, mag_preds)
                out['m_MAE_size'] = m_mae_size
            else:
                out['m_MAE'] = paddle.zeros(shape=[1])
                out['m_MAE_size'] = m_mae_size
        return out
