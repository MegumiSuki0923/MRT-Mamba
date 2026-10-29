import argparse
import json
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import MRTMamba
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
from utils.tools import EarlyStopping, adjust_learning_rate, vali

parser = argparse.ArgumentParser(description="MRT-Mamba")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


parser.add_argument(
    "--is_training", type=int, required=True, default=1, help="status", choices=(1,)
)
parser.add_argument(
    "--model_comment", type=str, required=True, default="none", help="prefix"
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="MRT-Mamba",
    help="model name",
    choices=("MRT-Mamba",),
)
parser.add_argument("--seed", type=int, default=2025, help="random seed")
parser.add_argument(
    "--data",
    type=str,
    required=True,
    default="ETTh1",
    help="dataset type",
    choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "Weather", "Traffic"),
)
parser.add_argument("--root_path", type=str, default="./dataset", help="root path")
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="M:multivariate, S:univariate, MS:mixed",
    choices=("M", "S", "MS"),
)
parser.add_argument("--target", type=str, default="OT", help="target feature")
parser.add_argument("--freq", type=str, default="h", help="freq for time features")
parser.add_argument(
    "--checkpoints", type=str, default="./checkpoints/", help="checkpoints location"
)
parser.add_argument(
    "--save_checkpoint",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Save the best validation checkpoint to disk; otherwise keep trainable weights in memory.",
)
parser.add_argument("--seq_len", type=int, default=512, help="input sequence length")
parser.add_argument("--label_len", type=int, default=48, help="start token length")
parser.add_argument("--pred_len", type=int, default=96, help="prediction length")
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--d_model", type=int, default=32, help="dimension of model")
parser.add_argument("--d_ff", type=int, default=128, help="dimension of fcn")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument("--embed", type=str, default="timeF", help="time features encoding")
parser.add_argument(
    "--small_patch", type=int, default=8, help="small branch patch length"
)
parser.add_argument("--small_stride", type=int, default=4, help="small branch stride")
parser.add_argument(
    "--large_patch", type=int, default=24, help="large branch patch length"
)
parser.add_argument("--large_stride", type=int, default=4, help="large branch stride")
parser.add_argument("--llm_layers", type=int, default=24, help="Mamba-130m: 24 layers")
parser.add_argument("--percent", type=int, default=100)
parser.add_argument(
    "--prompt_tokens",
    type=int,
    default=16,
    help="Total prompt tokens; the paper configuration uses 16 (4 meta, 6 pattern, 6 numerical).",
)
parser.add_argument(
    "--num_pattern_types",
    type=int,
    default=8,
    help="Number of pattern prototypes; the paper configuration uses 8.",
)
parser.add_argument(
    "--num_workers", type=int, default=2, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="Adam weight decay"
)
parser.add_argument(
    "--lradj",
    type=str,
    default="type1",
    help="adjust learning rate",
    choices=("type1", "type2", "type3", "PEMS", "TST", "constant", "COS"),
)
parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
parser.add_argument(
    "--clip_grad",
    type=float,
    default=1.0,
    help="maximum gradient norm for clipping, set to 0 to disable",
)
parser.add_argument(
    "--result_json",
    type=str,
    default="",
    help="optional path to save structured run metrics",
)
parser.add_argument(
    "--save_forecast_vis",
    action="store_true",
    default=False,
    help="save last-window forecast CSV after training",
)
parser.add_argument(
    "--save_pred",
    action="store_true",
    default=False,
    help="save test predictions, ground truth and inputs as npy",
)
parser.add_argument(
    "--save_pred_dir",
    type=str,
    default="./pred_results",
    help="directory to save test predictions",
)
args = parser.parse_args()
set_random_seed(args.seed)
script_start_time = time.time()
if args.batch_size <= 0:
    raise ValueError(f"[MRT-Mamba] batch_size must be positive, got {args.batch_size}")
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def format_time(seconds):
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    if m > 0:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"


run_summaries = []
for ii in range(args.itr):
    run_start_time = time.time()
    current_time = time.strftime("%Y%m%d%H%M%S")
    setting_name = f"{args.data}_{args.seq_len}_{args.pred_len}_{args.model_comment}_{current_time}"
    config_message = f"[Config] model={args.model} data={args.data} seq_len={args.seq_len} pred_len={args.pred_len} batch_size={args.batch_size} early_stop=val_loss patience={args.patience}"
    accelerator.print(config_message)
    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")
    accelerator.print(
        f"[Data] train={len(train_data)} val={len(vali_data)} test={len(test_data)} train_steps={len(train_loader)} val_steps={len(vali_loader)} test_steps={len(test_loader)}"
    )
    best_metrics = {
        "epoch": 0,
        "train_loss": None,
        "val_loss": float("inf"),
        "test_loss": None,
        "test_mae_loss": None,
    }
    run_status = "completed"
    nonfinite_reason = None
    model = MRTMamba.Model(args)
    if hasattr(model, "llm_model"):
        model.llm_model = model.llm_model.to(accelerator.device)
    path = os.path.join(args.checkpoints, setting_name)
    if (
        args.save_checkpoint
        and (not os.path.exists(path))
        and accelerator.is_local_main_process
    ):
        os.makedirs(path)
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    best_trainable_keys = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    best_state_cpu = None
    model_optim = optim.Adam(
        trained_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=args.train_epochs, eta_min=1e-08
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = (
        accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )
    )
    early_stopping = EarlyStopping(
        accelerator=accelerator, patience=args.patience, save_mode=args.save_checkpoint
    )
    epoch_times = []
    epochs_completed = 0
    for epoch in range(args.train_epochs):
        train_loss = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(train_loader)
        ):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            dec_inp = (
                torch.zeros_like(batch_y[:, -args.pred_len :, :])
                .float()
                .to(accelerator.device)
            )
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )
            with accelerator.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len :, f_dim:]
                batch_y = batch_y[:, -args.pred_len :, f_dim:].to(accelerator.device)
                loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                if args.clip_grad > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=args.clip_grad
                    )
            model_optim.step()
        epoch_cost_time = time.time() - epoch_time
        epoch_times.append(epoch_cost_time)
        if args.lradj == "COS":
            scheduler.step()
            accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
        elif args.lradj != "TST":
            if epoch == 0:
                args.learning_rate = model_optim.param_groups[0]["lr"]
                accelerator.print(
                    "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                )
            adjust_learning_rate(
                accelerator, model_optim, scheduler, epoch + 1, args, printout=True
            )
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = args.train_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        train_loss = np.average(train_loss)
        vali_loss, _ = vali(
            args, accelerator, model, vali_data, vali_loader, criterion, mae_metric
        )
        epochs_completed = epoch + 1
        if not np.isfinite(train_loss) or not np.isfinite(vali_loss):
            nonfinite_reason = f"nonfinite_eval_metrics epoch={epoch + 1}"
            run_status = "nonfinite_eval_metrics"
            message = f"[Abort] {nonfinite_reason} train={train_loss} val={vali_loss}"
            accelerator.print(message)
            break
        is_best_epoch = vali_loss < best_metrics["val_loss"]
        test_loss, test_mae_loss = vali(
            args, accelerator, model, test_data, test_loader, criterion, mae_metric
        )
        if not np.isfinite(test_loss) or not np.isfinite(test_mae_loss):
            nonfinite_reason = f"nonfinite_test_metrics epoch={epoch + 1}"
            run_status = "nonfinite_test_metrics"
            message = f"[Abort] {nonfinite_reason} test={test_loss} mae={test_mae_loss}"
            accelerator.print(message)
            break
        cost_message = "Epoch: {}/{} | Cost: {} | ETA: {}".format(
            epoch + 1,
            args.train_epochs,
            format_time(epoch_cost_time),
            format_time(estimated_remaining),
        )
        metric_message = "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss
        )
        accelerator.print(cost_message)
        accelerator.print(metric_message)
        if is_best_epoch:
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(vali_loss),
                "test_loss": float(test_loss),
                "test_mae_loss": float(test_mae_loss),
            }
            if not args.save_checkpoint:
                best_state_cpu = {
                    k: v.detach().clone().cpu()
                    for k, v in accelerator.unwrap_model(model).state_dict().items()
                    if k in best_trainable_keys
                }
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            run_status = "early_stopped"
            break
    if args.save_forecast_vis or args.save_pred:
        best_model_path = os.path.join(path, "checkpoint")
        if args.save_checkpoint and os.path.exists(best_model_path):
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(
                torch.load(best_model_path, map_location=accelerator.device)
            )
        elif not args.save_checkpoint and best_state_cpu:
            unwrapped_model = accelerator.unwrap_model(model)
            missing, unexpected = unwrapped_model.load_state_dict(
                best_state_cpu, strict=False
            )
            accelerator.print(
                f"[Export] Restored best-validation weights ({len(best_state_cpu)} trainable tensors), unexpected={len(unexpected)}"
            )
            del best_state_cpu
        model.eval()
        total_test_batches = len(test_loader)
        last_window_data = None
        f_dim = -1 if args.features == "MS" else 0
        all_preds = []
        all_trues = []
        all_inputs = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
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
                if args.save_pred:
                    pred_batch = (
                        outputs[:, -args.pred_len :, f_dim:].detach().cpu().numpy()
                    )
                    true_batch = (
                        batch_y[:, -args.pred_len :, f_dim:].detach().cpu().numpy()
                    )
                    input_batch = batch_x.detach().cpu().numpy()
                    all_preds.append(pred_batch)
                    all_trues.append(true_batch)
                    all_inputs.append(input_batch)
                if args.save_forecast_vis and i == total_test_batches - 1:
                    pred_slice = outputs[:, -args.pred_len :, f_dim:]
                    true_slice = batch_y[:, -args.pred_len :, f_dim:]
                    last_window_data = (
                        batch_x[-1].cpu().numpy(),
                        true_slice[-1].cpu().numpy(),
                        pred_slice[-1].cpu().detach().numpy(),
                    )
        if args.save_pred and accelerator.is_local_main_process:
            pred_arr = np.concatenate(all_preds, axis=0)
            true_arr = np.concatenate(all_trues, axis=0)
            input_arr = np.concatenate(all_inputs, axis=0)
            save_dir = os.path.join(args.save_pred_dir, setting_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "pred.npy"), pred_arr)
            np.save(os.path.join(save_dir, "true.npy"), true_arr)
            np.save(os.path.join(save_dir, "input.npy"), input_arr)
            accelerator.print(
                f"[Export] Saved pred={pred_arr.shape}, true={true_arr.shape}, input={input_arr.shape} to {save_dir}"
            )
        if (
            args.save_forecast_vis
            and accelerator.is_local_main_process
            and (last_window_data is not None)
        ):
            hist_arr, fut_true_arr, fut_pred_arr = last_window_data
            hist_vals = hist_arr[:, 0]
            true_vals = fut_true_arr[:, 0]
            pred_vals = fut_pred_arr[:, 0]
            vis_dir = "./forecast_vis"
            os.makedirs(vis_dir, exist_ok=True)
            csv_path = os.path.join(vis_dir, setting_name + ".csv")
            with open(csv_path, "w") as csv_f:
                csv_f.write("time_step,ground_truth,prediction\n")
                for t, v in enumerate(hist_vals):
                    csv_f.write(f"{t},{v:.6f},\n")
                offset = len(hist_vals)
                for t, (tv, pv) in enumerate(zip(true_vals, pred_vals)):
                    csv_f.write(f"{offset + t},{tv:.6f},{pv:.6f}\n")
            msg = f"[Export] Last test window saved: {csv_path}"
            accelerator.print(msg)
        model.train()
    run_total_time = time.time() - run_start_time
    run_summaries.append(
        {
            "run_index": ii + 1,
            "setting_name": setting_name,
            "best_epoch": best_metrics["epoch"],
            "best_train_loss": best_metrics["train_loss"],
            "best_val_loss": best_metrics["val_loss"],
            "best_test_loss": best_metrics["test_loss"],
            "best_test_mae_loss": best_metrics["test_mae_loss"],
            "epochs_completed": epochs_completed,
            "run_total_time_sec": float(run_total_time),
            "status": run_status,
            "nonfinite_reason": nonfinite_reason,
        }
    )
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Run total time: {format_time(run_total_time)}")
    accelerator.print(f"{'=' * 60}")
accelerator.wait_for_everyone()
script_total_time = time.time() - script_start_time
hours, remainder = divmod(script_total_time, 3600)
minutes, seconds = divmod(remainder, 60)
accelerator.print(f"\n{'=' * 60}")
accelerator.print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
accelerator.print(f"{'=' * 60}")
if args.result_json and accelerator.is_local_main_process:
    result_dir = os.path.dirname(args.result_json)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    result_payload = {
        "model": args.model,
        "data": args.data,
        "objective": "minimize_best_val_loss",
        "script_total_time_sec": float(script_total_time),
        "args": vars(args),
        "runs": run_summaries,
    }
    if run_summaries:
        result_payload["best_run"] = min(
            run_summaries, key=lambda item: item["best_val_loss"]
        )
    with open(args.result_json, "w", encoding="utf-8") as result_file:
        json.dump(result_payload, result_file, ensure_ascii=False, indent=2)
