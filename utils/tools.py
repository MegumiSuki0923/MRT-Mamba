import numpy as np
import torch
from tqdm import tqdm


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * 0.5 ** ((epoch - 1) // 1)}
    elif args.lradj == "type2":
        lr_adjust = {
            2: 5e-05,
            4: 1e-05,
            6: 5e-06,
            8: 1e-06,
            10: 5e-07,
            15: 1e-07,
            20: 5e-08,
        }
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 3
                else args.learning_rate * 0.9 ** ((epoch - 3) // 1)
            )
        }
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * 0.95 ** (epoch // 1)}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            if accelerator is not None:
                accelerator.print("Updating learning rate to {}".format(lr))
            else:
                print("Updating learning rate to {}".format(lr))


class EarlyStopping:

    def __init__(
        self,
        accelerator=None,
        patience=7,
        verbose=False,
        delta=0,
        save_mode=True,
        logger=None,
    ):
        if not isinstance(save_mode, bool):
            raise ValueError("save_mode must be a boolean")
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode
        self.logger = logger

    def _log(self, message):
        if self.logger is not None:
            self.logger(message)
        elif self.accelerator is None:
            print(message)
        else:
            self.accelerator.print(message)

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self._log(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        return None

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self._log(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
        torch.save(model.state_dict(), path + "/checkpoint")
        self.val_loss_min = val_loss


def vali(
    args,
    accelerator,
    model,
    vali_data,
    vali_loader,
    criterion,
    mae_metric,
    show_progress=True,
):
    total_loss = []
    total_mae_loss = []
    total_samples = 0
    model.eval()
    with torch.no_grad():
        progress = tqdm(
            vali_loader,
            disable=not show_progress or not accelerator.is_local_main_process,
            leave=False,
        )
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(progress):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )
            with accelerator.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:].to(accelerator.device)
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            n_samples = pred.shape[0]
            total_loss.append(loss.item() * n_samples)
            total_mae_loss.append(mae_loss.item() * n_samples)
            total_samples += n_samples
    total_loss = np.sum(total_loss) / total_samples
    total_mae_loss = np.sum(total_mae_loss) / total_samples
    model.train()
    return (total_loss, total_mae_loss)
