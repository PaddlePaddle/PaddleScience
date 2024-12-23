import os
from dataset.dataset import get_datalist


def get_samples(root):
    folds = [f'param{i}' for i in range(9)]
    samples = []
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold))
        for file in files:
            path = os.path.join(root, os.path.join(fold, file))
            if os.path.isdir(path):
                fold_samples.append(os.path.join(fold, file))
        samples.append(fold_samples)
    return samples


def load_train_val_fold(args, preprocessed):
    samples = get_samples(args.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == args.fold_id:
            continue
        trainlst += samples[i]
    vallst = samples[args.fold_id] if 0 <= args.fold_id < len(samples
        ) else None
    if preprocessed:
        print('use preprocessed data')
    print('loading data')
    train_dataset, coef_norm = get_datalist(args.data_dir, trainlst, norm=
        True, savedir=args.save_dir, preprocessed=preprocessed)
    val_dataset = get_datalist(args.data_dir, vallst, coef_norm=coef_norm,
        savedir=args.save_dir, preprocessed=preprocessed)
    print('load data finish')
    return train_dataset, val_dataset, coef_norm


def load_train_val_fold_file(args, preprocessed):
    samples = get_samples(args.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == args.fold_id:
            continue
        trainlst += samples[i]
    vallst = samples[args.fold_id] if 0 <= args.fold_id < len(samples
        ) else None
    if preprocessed:
        print('use preprocessed data')
    print('loading data')
    train_dataset, coef_norm = get_datalist(args.data_dir, trainlst, norm=
        True, savedir=args.save_dir, preprocessed=preprocessed)
    val_dataset = get_datalist(args.data_dir, vallst, coef_norm=coef_norm,
        savedir=args.save_dir, preprocessed=preprocessed)
    print('load data finish')
    return train_dataset, val_dataset, coef_norm, vallst
