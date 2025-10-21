import torch
from training import train_nn
import config as cfg
import matplotlib.pyplot as plt


def run():
    best_model, loss_history, runtime, latest_model = train_nn(cfg.XL, cfg.XR, cfg.BOUNDARY_CONDITION, cfg.HIDDEN_DIM, cfg.EPOCHS, cfg.NUM_GRID_POINTS, cfg.LR, cfg.MINIBATCH_NUMBER)

    print('Training time (minutes):', runtime/60)
    plt.figure(figsize = (8,6));
    plt.loglog(loss_history[0],'-b',alpha=0.975);
    plt.ylabel('Total Loss');
    plt.xlabel('Epochs');
    plt.tight_layout()

    energy_loss_cpu = [loss.cpu().detach().numpy() if torch.is_tensor(loss) else loss for loss in loss_history[8]]
    plt.figure(figsize = (8,6))
    plt.plot(energy_loss_cpu);
    plt.ylabel('Model Energy History');
    plt.xlabel('Epochs');
    plt.tight_layout()


if __name__ == '__main__':
    run()
