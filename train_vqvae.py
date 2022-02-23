import argparse
from accelerate import Accelerator, notebook_launcher

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from torchinfo import summary
import wandb

from tqdm.auto import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler


def train(loader, val_loader):
    accelerator = Accelerator(fp16=False, cpu=args.cpu_run)
    device = accelerator.device

    #initializing the model
    model = VQVAE(in_channel=3, channel=128, n_res_block=args.res_blocks,
                  n_res_channel=args.res_channel,
                  embed_dim=args.embed_dim, n_embed=args.n_embed,
                  decay=args.decay).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None

    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    accelerator.print(summary(model, (args.batch_size, 3, args.size, args.size)))
    accelerator.print(model) #vanilla pytorch summary

    model, optimizer, loader, val_loader = accelerator.prepare(model, optimizer, loader, val_loader)
    loader, val_loader = tqdm(loader), tqdm(val_loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    latent_loss_beta_list = torch.linspace(0, latent_loss_weight, 20)

    sample_size = 20

    mse_sum = 0
    mse_n = 0
    val_mse_sum, val_mse_n, beta_index = 0, 0, 0 #init beta index

    with wandb.init(project=args.wandb_project_name, config=args.__dict__, save_code=True, name=args.run_name, magic=True): 
        for epoch in range(args.epoch):
            #Starting Epoch loops
            model.train()
            for i, (img, label) in enumerate(loader):
                model.zero_grad(set_to_none=True)

                img = img.to(device)

                out, latent_loss = model(img)

                with accelerator.autocast():
                    recon_loss = criterion(out, img)

                latent_loss = latent_loss.mean()

                beta_index = epoch  #for consistency
                if beta_index >= 24:
                    beta_index = latent_loss_beta_list[-1]

                loss = recon_loss + latent_loss_beta_list[beta_index] * latent_loss

                accelerator.backward(loss) #added loss to backprop

                if scheduler is not None:
                    scheduler.step()

                accelerator.clip_grad_norm_(model.parameters(), args.gradclip) #grad clipping

                optimizer.step()

                mse_sum += recon_loss.item() * img.shape[0]
                mse_n += img.shape[0]

                lr = optimizer.param_groups[0]['lr']

                wandb.log({"epoch": epoch+1, "mse": recon_loss.item(), 
                            "latent_loss": latent_loss.item(), "avg_mse": (mse_sum / mse_n), 
                            "lr": lr})

                if i % 25 == 0:
                    accelerator.print({"epoch": epoch+1, "mse": recon_loss.item(),
                        "latent_loss": latent_loss.item(), "avg_mse": (mse_sum/ mse_n), 
                        "lr": lr})

                #Performing Validation and loggign out images
                if epoch % 2 == 0:   #i % 100 == 0
                    model.eval()

                    #--------------VALIDATION------------------
                    for i, (img, label) in enumerate(val_loader):
                        img = img.to(device)
                        mode = model.to(device)

                        with torch.no_grad():
                            out, latent_loss = model(img)

                        val_recon_loss = criterion(out, img)
                        val_latent_loss = latent_loss.mean()
                        val_loss = recon_loss + latent_loss_beta_list[beta_index] * latent_loss

                        val_mse_sum += recon_loss.item() * img.shape[0]
                        val_mse_n += img.shape[0]

                    wandb.log({"epoch": epoch+1, "val_mse": val_recon_loss.item(), 
                            "val_latent_loss": val_latent_loss.item(), "val_avg_mse": (val_mse_sum/ val_mse_n), 
                            "lr": lr})

                    accelerator.print({"epoch": epoch+1, "val_mse": val_recon_loss.item(), 
                            "val_latent_loss": val_latent_loss.item(), "val_avg_mse": (val_mse_sum/ val_mse_n), 
                            "lr": lr})

                    model.train()

            #Saving the model checkpoints every epoch
            if epoch % 10 == 0:
                model.eval()
                
                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f'/kaggle/working/samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                wandb.log({f"{epoch+1}_Samples" : [wandb.Image(img) for img in torch.cat( [sample, out], 0) ]})

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), f'checkpoint/vqvae_{str(epoch + 1).zfill(3)}.pt')
                model.train()

            print(f'\n---EPOCH {epoch} CCOMPLETED---\n')

if __name__ == '__main__':
    '''
    These are the default values:-
    n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-blocks', type=int, default=4)
    parser.add_argument('--res-channel', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--n-embed', type=int, default=512)
    parser.add_argument('--gradclip', type=float, default=5)
    parser.add_argument('--decay', type=float, default=0.99)

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--wandb-project-name', type=str)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--cpu-run', type=bool, default=False)
    parser.add_argument('training_path', type=str)
    parser.add_argument('--validation-path', type=str)

    args = parser.parse_args()

    print(args)

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.training_path, transform=transform)
    val_dataset = datasets.ImageFolder(args.validation_path, transform=transform)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, #True
                        num_workers=args.num_workers, pin_memory=True, prefetch_factor=32)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, prefetch_factor=32)

    #Finally starting the training
    notebook_launcher(train(loader, val_loader))