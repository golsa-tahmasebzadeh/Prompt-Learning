import sys
# from tkinter.tix import Tree
sys.path.insert(1, '/nfs/home/tahmasebzadehg/prompt_learning/src')

import logging
import os
import random
from datetime import datetime
import json
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

import pandas as pd



try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model
from training.data_coop import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train_coop import train_one_epoch, evaluate

# me
from open_clip.coop_model import COOPCLIP
import gc
gc.collect()
torch.cuda.empty_cache()
#


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args, n_shot_in, n_round, device):
# add val dataset for all experiments vise_val_100_shot_round_1.csv
    if args.dataset_name == 'vise':  t = 'vise_'
    else: t = ''
    args.shot = n_shot_in
    args.train_data = f'{args.data_dir}/{args.dataset_name}/train/round{n_round}/{t}train_{n_shot_in}_shot_round_{n_round}.csv'
    args.val_data =  f'{args.data_dir}/{args.dataset_name}/val/round{n_round}/{t}val_{n_shot_in}_shot_round_{n_round}.csv'
    if args.dataset_name == 'instances':
        args.val_data = None

    args.kg_info_path = f'{args.data_dir}/{args.dataset_name}/kg_info.json'
    args.class_names_path = f'{args.data_dir}/{args.dataset_name}/class_names.csv'

    args.name =  f'{args.pre_fname}-init-{args.kg_init}-model-{args.model}-ctx-{args.N_CTX}-{args.CLASS_TOKEN_POSITION}-shot-{args.shot}-round-{n_round}'
    # me    

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path):
        #     print( "Error. Experiment already exists. Use --name {} to specify a new experiment."  )
        #     return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    
    
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ['amp', 'amp_bfloat16', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    clip_model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
    )

    ##### load kg info if exists
    kg_info = None
    if os.path.exists(args.kg_info_path):
        with open(args.kg_info_path, encoding='utf8') as f_kg:
            kg_info = json.load(f_kg)

    classnames = pd.read_csv(args.class_names_path)[args.gt_label_name]
    model = COOPCLIP( list( classnames ), kg_info , clip_model,  device)

    # train prompt learner - freeze backbones
    if args.no_train_backbone == True:
            for name, p in clip_model.named_parameters():  
                  p.requires_grad = False   

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable Parameters backbone: {n_params}")

    random_seed(args.seed, args.rank)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                # print(model)
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))


    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val, classnames), epoch=start_epoch)
    #me
    data['classnames'] = classnames
    args.warmup = int( (len(classnames) * args.shot) /args.batch_size)
    #me
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    n = data["train"].dataloader.num_samples
    if 'train' in data and optimizer is not None:
        
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    # if args.save_logs and args.tensorboard:
    assert tensorboard is not None, "Please install tensorboard."
    # writer = tensorboard.SummaryWriter('../../sbatch/logs/00tensorboard')

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    best_val_loss = 10e4
    best_val_acc = 0
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            val_metrics = evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {  "epoch": completed_epoch,  "name": args.name, "state_dict": model.state_dict(),"optimizer": optimizer.state_dict(),  }

            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if args.dataset_name == 'instances': # if there is no kg info file
                if completed_epoch == args.epochs or (  args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0  ):
                    torch.save( checkpoint_dict,  os.path.join(args.checkpoint_path, f"chk.pt"),   )
            else:
                if val_metrics['@1'] >= best_val_acc: 
                    best_val_acc = val_metrics['@1']
                    # torch.save( checkpoint_dict,  os.path.join(args.checkpoint_path + '/acc_' + str( best_val_acc ) + f"_epoch_{completed_epoch}.pt"),   )  
                    torch.save( checkpoint_dict,  os.path.join(args.checkpoint_path + "/chk.pt"),   )    
                elif completed_epoch == args.epochs:
                    torch.save( checkpoint_dict,  os.path.join(args.checkpoint_path + "/chk.pt"),   ) 
                    # torch.save( checkpoint_dict,  os.path.join(args.checkpoint_path + '/acc_' + str( val_metrics['@1'] ) + f"_epoch_{completed_epoch}.pt"),   ) 

    if args.wandb and is_master(args):
        wandb.finish()
    # print('done!')
    return device


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    # if os.path.exists(new_code_path):
    #     print(
    #         f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
    #     )
    #     return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    args = parse_args()
    device = None
    args.distribute_device = 'True'
    shots = [5] #[ 1,2,3,4,5,10,20,30,50 ]
        
    for _shot in shots:
            device = main(args, _shot, args.data_set_number, device)
            args.distribute_device = 'False'


