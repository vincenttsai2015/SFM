import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Dirichlet

from models import get_flow_model
from datasets import get_dataset
from utils import seed_all, load_config, get_optimizer, get_scheduler, count_parameters, recursive_to_device
from visualize import get_vis
from evaluation.fid import get_fid, calculate_activation_statistics, InceptionV3

def cal_elbo(model, dataset, max_sample=1000, batch_size=100, method='ode', n_step=500, tmax=0.995, device='cuda'):
    generator = torch.Generator()
    generator.manual_seed(42)
    idx = torch.randperm(len(dataset), generator=generator)[:max_sample]
    subset = [dataset[i][0] for i in idx]

    nlls = []
    model.eval()
    with torch.no_grad():
        for i in range(0, max_sample, batch_size):
            x = torch.stack(subset[i:i + batch_size], dim=0).to(device)
            nll = model.compute_elbo(method, x.to(device), n_step, tmax=tmax, verbose=True)
            nlls.append(nll.item())
            print(f'NLL: {sum(nlls) / len(nlls):.4f}')
    nlls = np.array(nlls)
    print(f'Avg NLL: {nlls.mean():.4f} ± {nlls.std():.4f}')

def toy_data_entropy(probs, seq_len, eps=1e-12):
    probs = torch.as_tensor(probs, dtype=torch.float32)
    probs = probs / probs.sum()
    h = -(probs * (probs + eps).log()).sum()
    return seq_len * h

@torch.no_grad()
def estimate_toy_kl(model, dataloader, probs, seq_len, method='ode', n_steps=200, tmax=0.995, device='cuda'):
    model.eval()
    H_data = toy_data_entropy(probs, seq_len).to(device)

    total_elbo_nll = 0.0
    total_count = 0

    total_sz = len(dataloader)

    for i, batch in tqdm(enumerate(dataloader), total=total_sz):
        batch = batch.to(device)   # shape: (B, seq_len, simplex_dim)

        # one-hot data on simplex boundary -> use ELBO, not direct NLL
        elbo_nll = model.compute_elbo(
            method=method,
            p1=batch,
            n_steps=n_steps,
            tmax=tmax,
            verbose=False
        )

        bsz = batch.size(0)
        total_elbo_nll += elbo_nll.item() * bsz
        total_count += bsz

    avg_elbo_nll = total_elbo_nll / total_count
    kl_est = avg_elbo_nll - H_data.item()
    return {
        "avg_elbo_nll": avg_elbo_nll,
        "data_entropy": H_data.item(),
        "kl_data_model": kl_est,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    visualizer = get_vis(config.visualizer, writer, args.device)

    # Data
    print('Loading datasets...')
    train_set, valid_set, test_set = get_dataset(config.datasets)

    # Dataloader
    if config.datasets.type == 'toy_dfm':
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size,)
        valid_loader = DataLoader(valid_set, batch_size=config.train.batch_size)
        test_loader = DataLoader(test_set, batch_size=config.train.batch_size)
    else:
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=16)
        valid_loader = DataLoader(valid_set, batch_size=config.train.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False, num_workers=8)

    # Model
    print('Building model...')
    model = get_flow_model(config.model, config.encoder).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
    global_step = 0


    def train():
        global global_step
        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for batch in train_loader:
                if config.conditioned:
                    x, *cond_args = batch
                else:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                        cond_args = batch[1:]
                    else:
                        x = batch
                        cond_args = []
                # Training
                x = x.to(args.device)
                cond_args = recursive_to_device(cond_args, args.device)
                loss = model.get_loss(x, *cond_args)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1

                # Validation
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate(valid_loader)
                    sample('euler', 'valid', config.train.batch_size)
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    torch.save({
                        'model': model.state_dict(),
                        'step': global_step,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate(dataloader, split='valid'):
        with torch.no_grad():
            model.eval()

            val_losses = []
            total = config.get('valid_max_batch', None)
            if total is None:
                total = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=total):
                if i >= total:
                    break

                # ✅ dataloader 可能回傳 (x, cond1, cond2, ...) 或只回傳 x
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    cond_args = batch[1:]
                else:
                    x = batch
                    cond_args = ()

                x = x.to(args.device)
                cond_args = recursive_to_device(cond_args, args.device)
                loss = model.get_loss(x, *cond_args)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar(f'{split}/loss', val_loss, global_step)
        print(f'Step {global_step} {split} loss {val_loss:.6f}')
        return val_loss

    def sample(method='euler', split='valid', max_batch=config.train.batch_size):
        with torch.no_grad():
            model.eval()
            if not config.conditioned:
                traj = visualizer(model, method, global_step)
            else:
                dataloader = valid_loader if split == 'valid' else test_loader
                traj = visualizer(model, dataloader, method, global_step, max_batch=max_batch)
        return traj
    
    try:
        if args.mode == 'train':
            train()
            print('Training finished!')
            sample('ode', 'valid', None)
            print('Sampling finished!')
        elif args.mode == 'inf': 
            if args.resume is None: 
                print('[WARNING]: inference mode without loading a pretrained model')
                
            elif config.datasets.type == 'bmnist':
                print('Loading model for inference...')
                ckpt = torch.load(args.resume, map_location=args.device)
                model.load_state_dict(ckpt['model'])
                model.eval()
                print('Model loading complete!')
                samples = []
                total_sample = len(test_set)
                n_batch = total_sample // config.train.batch_size
                with torch.no_grad():
                    for _ in tqdm(range(n_batch)):
                        s = model.sample('ode', n_sample=config.train.batch_size, n_steps=300, device=args.device)
                        samples.append(s.detach().cpu())
                    samples = torch.cat(samples)
                    img = (samples[..., 0] > samples[..., 1]).float().view(-1, 1, 28, 28).expand(-1, 3, -1, -1)
                    print(f'Generated samples shape: {img.shape}')
                # sample('ode', 'test', None)
                print('Sampling finished!')
                
                print('Calculating FID...')
                gt = []
                for samples, *_ in tqdm(test_loader):
                    img = (samples[..., 0] > samples[..., 1]).float().view(-1, 1, 28, 28).expand(-1, 3, -1, -1)
                    gt.append(img)
                gt = torch.cat(gt, dim=0)
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                inception = InceptionV3([block_idx]).to(args.device)
                inception.eval()
                mu, sigma = calculate_activation_statistics(gt, inception, args.device, batch_size=256)
                np.savez('bmnist_fid.npz', mu=mu, sigma=sigma)
                print('Done! Statistics saved to bmnist_fid.npz')

                fid = get_fid(img, args.device, batch_size=config.train.batch_size)
                print(f'FID: {fid:.4f}')
                cal_elbo(model, test_set, max_sample=total_sample, batch_size=config.train.batch_size, method='ode', n_step=200, tmax=0.9, device=args.device)
            
            elif config.datasets.type == 'toy_dfm':
                ckpt = torch.load(args.resume, map_location=args.device)
                model.load_state_dict(ckpt['model'])
                model.eval()
                print('Model loading complete!')
                stats = estimate_toy_kl(
                    model=model,
                    dataloader=test_loader,
                    probs=test_loader.dataset.probs,
                    seq_len=test_loader.dataset.seq_len,
                    method='ode',
                    n_steps=200,
                    tmax=0.99,
                    device=args.device,
                )

                print("ELBO-NLL:", stats["avg_elbo_nll"])
                print("H(data):", stats["data_entropy"])
                print("KL(data || model):", stats["kl_data_model"])

            time.sleep(3)  # Wait for the last tensorboard logs to be written
        else:
            print('Please choose either train or inf.')
    except KeyboardInterrupt:
        print('Terminating...')
