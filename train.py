import torch
from tqdm import tqdm
import wandb
import utils.utils as utils

def train_discriminator(imgs, discriminator, criterion, optimizer, scaler, label, device):
    optimizer.zero_grad()
    imgs, discriminator = imgs.to(device, non_blocking = True), discriminator.to(device)
    labels = torch.full((len(imgs),), label, dtype=torch.float).to(device)
    with torch.cuda.amp.autocast():
        out = discriminator(imgs).view(-1)
        batch_loss = criterion(out, labels)
    scaler.scale(batch_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return batch_loss.item()

def train(dataloader, generator, discriminator, num_epochs, base_criterion, gen_opt, disc_opt, latent_dim,
          scheduler, diversity_metric, div_lambda = 0.1, load_checkpoint = False, scheduler_flag = False, 
          div_flag = False, gen_save = 'saved_models/generator.pt', device = 'cpu',
          disc_save = 'saved_models/discriminator.pt'):
    if load_checkpoint:
        generator = utils.load_checkpoint(generator, gen_save)
        discriminator = utils.load_checkpoint(discriminator, disc_save)
    

    diversities = []
    generator_losses, discriminator_losses = [], [] 
    fake_label, real_label = 0, 1
    iteration = 1

    scaler = torch.cuda.amp.GradScaler()
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    for epoch in range(num_epochs+1):
        epoch_disc_loss, epoch_gen_loss, epoch_div_metric = 0.0, 0.0, 0.0
        discriminator.train()
        generator.train()
        progress_bar = tqdm(dataloader, total = len(dataloader), leave = False)
        for imgs in progress_bar:
            # Training Discriminator
            iter_real_losses = train_discriminator(imgs, discriminator, base_criterion, disc_opt, 
                                                       scaler, real_label, device=device)
            fake_samples = torch.randn((len(imgs), latent_dim)).to(device)
            fake_imgs = generator(fake_samples)
            iter_fake_losses = train_discriminator(fake_imgs.detach(), discriminator, base_criterion, disc_opt,
                                                   scaler, fake_label, device = device)
            epoch_disc_loss += (iter_real_losses + iter_fake_losses)/2.0
            
            # Training Generator
            gen_opt.zero_grad()
            real_labels = torch.full((len(imgs),), real_label, dtype=torch.float).to(device)
            with torch.cuda.amp.autocast():
                out = discriminator(fake_imgs).view(-1)
                iter_gen_loss = base_criterion(out, real_labels)
            if div_flag:
                diversity = diversity_metric(fake_imgs, device = device)
                final_loss = iter_gen_loss - div_lambda*diversity
            else:
                final_loss = iter_gen_loss
            scaler.scale(final_loss).backward()
            scaler.step(gen_opt)
            scaler.update()

            # Calculating diversity in the generated image
            if not(div_flag):
                diversity = diversity_metric(fake_imgs, device = device)
            epoch_gen_loss += iter_gen_loss.item()
            epoch_div_metric += diversity.item()

            # Storing all losses and logs
            progress_bar.set_postfix(disc_loss = (iter_fake_losses + iter_real_losses)/2.0, gen_loss = iter_gen_loss.item(), 
                                     diversity = diversity.item())
            wandb.log({'iterations': iteration, 'disc_loss': (iter_fake_losses + iter_real_losses)/2.0, 
                       'gen_loss': iter_gen_loss.item(), 'diversity': diversity.item()})
            iteration+=1
        
        epoch_disc_loss /= len(dataloader)
        epoch_gen_loss /= len(dataloader)
        epoch_div_metric /= len(dataloader)

        generator_losses.append(epoch_gen_loss)
        discriminator_losses.append(epoch_disc_loss)
        diversities.append(epoch_div_metric)
        wandb.log({'epoch':epoch ,'generator_losses': epoch_gen_loss, 'discriminator_losses': epoch_disc_loss, 
                    'diversity_metric':epoch_div_metric})
        utils.visualization(generator, "Epoch_train", latent_dim = latent_dim, device = device)
        print(f"{'*'*10} EPOCH {epoch:2}/{num_epochs} {'*'*10}")
        print(f'''{"#"*33}
Diversity Score: {epoch_div_metric:5.3f}
Discriminator Loss: {epoch_disc_loss:5.3f}
Generator Loss: {epoch_gen_loss:5.3f}
{"#"*33}''')
        utils.save_checkpoint(discriminator, disc_save)
        utils.save_checkpoint(generator, gen_save)
        if scheduler_flag:
            scheduler.step()
    return generator_losses, discriminator_losses, diversities