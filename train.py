import os, datetime, time
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import (
    calc_loss, get_model, get_train_loader, inference, merge_config,
    save_model, cp_projects, get_work_dir, get_logger
)
from evaluation.eval_wrapper import eval_lane


# =========================
#   DEBUG HELPERS
# =========================
def debug_batch(data_label, cfg):
    """Visualiza un batch y muestra stats de las labels (proyectadas sobre la imagen)."""
    print("\n=== DEBUG DATA_LABEL ===")
    for k, v in data_label.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={v.min().item()}, max={v.max().item()}")
        else:
            print(f"{k}: type={type(v)}")

    img_tensor = data_label["images"][0]  # [3,H,W]
    labels_row = data_label["labels_row"][0].cpu().numpy()  # (num_row, num_lanes)
    labels_col = data_label["labels_col"][0].cpu().numpy()  # (num_col, num_lanes)
    row_anchor = cfg.row_anchor
    col_anchor = cfg.col_anchor

    # Des-normaliza imagen
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * std + mean, 0, 1)

    H_img, W_img = img.shape[:2]

    plt.figure(figsize=(16, 4))
    plt.imshow(img)

    # Row-anchors (rojo): fijamos Y (ra), pintamos X desde labels_row
    # NOTA: row_anchor y col_anchor están en [0..1] (merge_config); no son píxeles.
    for lane_idx in range(labels_row.shape[1]):
        xs, ys = [], []
        for r, ra in enumerate(row_anchor):
            col_id = labels_row[r, lane_idx]
            if col_id != -1:
                x = int(col_id / cfg.num_cell_col * W_img)
                y = int(ra * H_img)  # ra ya es fracción de altura
                xs.append(x); ys.append(y)
        if len(xs) > 1:
            plt.plot(xs, ys, "ro-", markersize=2, linewidth=1)

    # Col-anchors (azul): fijamos X (ca), pintamos Y desde labels_col
    for lane_idx in range(labels_col.shape[1]):
        xs, ys = [], []
        for c, ca in enumerate(col_anchor):
            row_id = labels_col[c, lane_idx]
            if row_id != -1:
                x = int(ca * W_img)  # ca es fracción de anchura
                y = int(row_id / cfg.num_cell_row * H_img)
                xs.append(x); ys.append(y)
        if len(xs) > 1:
            plt.plot(xs, ys, "bo-", markersize=2, linewidth=1)

    plt.axis("off")
    plt.title("INPUT + labels proyectados (row=rojo, col=azul)")
    plt.show()

    valid_ratio_row = (data_label["labels_row"] != -1).float().mean().item()
    valid_ratio_col = (data_label["labels_col"] != -1).float().mean().item()
    print(f"Valid ratio row={valid_ratio_row:.3f}, col={valid_ratio_col:.3f}")


def overfit_one_batch(net, train_loader, cfg, loss_dict, metric_dict, logger, iters=200, plot_pred=True):
    """Intenta sobreajustar un solo batch y muestra pérdidas/metrics periódicamente."""
    print("\n[DEBUG] Overfit en un solo batch...\n")
    net.train()
    data_iter = iter(train_loader)

    # Busca un batch con al menos algún carril positivo
    while True:
        data_label = next(data_iter)
        has_pos = (data_label['labels_row'] != -1).any() or (data_label['labels_col'] != -1).any()
        if has_pos:
            break

    # Visualiza y muestra stats
    debug_batch(data_label, cfg)

    # Prepara en GPU
    labels = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data_label.items()}
    optimizer = get_optimizer(net, cfg)

    for step in range(iters):
        results = inference(net, labels, cfg.dataset)
        loss = calc_loss(loss_dict, results, logger, step, 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"[Iter {step:03d}] Total Loss = {loss.item():.4f}")
            # Desglosa cada componente de loss
            for name, op, w, src in zip(loss_dict['name'],
                                        loss_dict['op'],
                                        loss_dict['weight'],
                                        loss_dict['data_src']):
                if w != 0:
                    datas = [results[s] for s in src]
                    loss_cur = op(*datas)
                    print(f"   {name}: {loss_cur.item():.4f} (w={w})")

            # Métricas originales
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                print(f"   {me_name}: {me_op.get():.3f}")

            # ===== Accuracies ENMASCARADAS (sólo anchors válidos) =====
     # ===== Accuracies ENMASCARADAS (sólo anchors válidos) =====
            with torch.no_grad():
                logits_row = results['cls_out']        # pred de filas
                logits_col = results['cls_out_col']    # pred de columnas

                gt_row = results['cls_label']          # (B, R, L)
                gt_col = results['cls_label_col']      # (B, C, L)

                # --- Row: localizar eje de clases (≈ num_cell_col [+1]) y hacer argmax ahí ---
                axis_row_class = None
                for ax, sz in enumerate(logits_row.shape):
                    if sz in (cfg.num_cell_col, cfg.num_cell_col + 1):
                        axis_row_class = ax
                        break
                assert axis_row_class is not None, f"No encuentro eje de clases en cls_out con shape {tuple(logits_row.shape)}"

                pred_row = logits_row.argmax(dim=axis_row_class)  # col_id por anchor
                # pred_row ahora es (B, *, *). Reordena a (B, L, R) para alinear con gt_row.permute(0,2,1)
                # Casos más comunes en UFLDv2:
                #   - (B, L, R)        -> OK
                #   - (B, R, L)        -> permutar
                #   - (B, R) (sin L)   -> improbable; lanza assert
                B, R, L = gt_row.shape[0], gt_row.shape[1], gt_row.shape[2]
                if pred_row.ndim != 3:
                    raise RuntimeError(f"pred_row ndim inesperado: {pred_row.ndim}, shape={tuple(pred_row.shape)}")

                if pred_row.shape[1:] == (L, R):
                    pass  # (B, L, R) ya correcto
                elif pred_row.shape[1:] == (R, L):
                    pred_row = pred_row.permute(0, 2, 1)  # -> (B, L, R)
                else:
                    # heurística: ordena ejes restantes por coincidencia con (R,L)
                    axes = list(pred_row.shape[1:])
                    msg = f"Forma pred_row no coincide. pred={tuple(pred_row.shape)}, gt_row(B,R,L)=({B},{R},{L})"
                    raise RuntimeError(msg)

                gt_row_BLR = gt_row.permute(0, 2, 1)  # (B, L, R)
                mask_row = gt_row_BLR != -1
                acc_row = (pred_row[mask_row] == gt_row_BLR[mask_row]).float().mean().item() if mask_row.any() else float('nan')
                vr_row = mask_row.float().mean().item()

                # --- Col: localizar eje de clases (≈ num_cell_row [+1]) y hacer argmax ahí ---
                axis_col_class = None
                for ax, sz in enumerate(logits_col.shape):
                    if sz in (cfg.num_cell_row, cfg.num_cell_row + 1):
                        axis_col_class = ax
                        break
                assert axis_col_class is not None, f"No encuentro eje de clases en cls_out_col con shape {tuple(logits_col.shape)}"

                pred_col = logits_col.argmax(dim=axis_col_class)  # row_id por anchor
                # Reordena a (B, L, C) para alinear con gt_col.permute(0,2,1)
                C = gt_col.shape[1]
                if pred_col.ndim != 3:
                    raise RuntimeError(f"pred_col ndim inesperado: {pred_col.ndim}, shape={tuple(pred_col.shape)}")

                if pred_col.shape[1:] == (cfg.num_lanes, C):          # (B, L, C)
                    pass
                elif pred_col.shape[1:] == (C, cfg.num_lanes):        # (B, C, L)
                    pred_col = pred_col.permute(0, 2, 1)              # -> (B, L, C)
                else:
                    msg = f"Forma pred_col no coincide. pred={tuple(pred_col.shape)}, gt_col(B,C,L)=({B},{C},{cfg.num_lanes})"
                    raise RuntimeError(msg)

                gt_col_BLC = gt_col.permute(0, 2, 1)  # (B, L, C)
                mask_col = gt_col_BLC != -1
                acc_col = (pred_col[mask_col] == gt_col_BLC[mask_col]).float().mean().item() if mask_col.any() else float('nan')
                vr_col = mask_col.float().mean().item()

            print(f"   MASKED acc_row: {acc_row:.3f}  (valid_ratio_row={vr_row:.3f})")
            print(f"   MASKED acc_col: {acc_col:.3f}  (valid_ratio_col={vr_col:.3f})")
            # ===== Plot pred vs GT (opcional) =====
            if plot_pred and step % 50 == 0:
                img_tensor = labels["images"][0].detach().cpu()  # [3,H,W]
                # desnormaliza
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img_tensor.numpy().transpose(1, 2, 0)
                img = np.clip(img * std + mean, 0, 1)
                H_img, W_img = img.shape[:2]

                labels_row = labels["labels_row"][0].detach().cpu().numpy()  # (R, L)
                labels_col = labels["labels_col"][0].detach().cpu().numpy()  # (C, L)
                row_anchor = cfg.row_anchor
                col_anchor = cfg.col_anchor

                pred_row_np = pred_row[0].detach().cpu().numpy()  # (L, R)
                pred_col_np = pred_col[0].detach().cpu().numpy()  # (L, C)

                plt.figure(figsize=(16, 4))
                plt.imshow(img)

                # ROW: GT rojo, PRED verde
                for lane_idx in range(labels_row.shape[1]):
                    xs, ys = [], []
                    for r, ra in enumerate(row_anchor):
                        col_id = labels_row[r, lane_idx]
                        if col_id != -1:
                            x = int(col_id / cfg.num_cell_col * W_img)
                            y = int(ra * H_img)
                            xs.append(x); ys.append(y)
                    if len(xs) > 1:
                        plt.plot(xs, ys, 'r.-', markersize=2, linewidth=1)

                for lane_idx in range(pred_row_np.shape[0]):
                    xs, ys = [], []
                    for r, ra in enumerate(row_anchor):
                        x = int(pred_row_np[lane_idx, r] / cfg.num_cell_col * W_img)
                        y = int(ra * H_img)
                        xs.append(x); ys.append(y)
                    if len(xs) > 1:
                        plt.plot(xs, ys, 'g.-', markersize=2, linewidth=1)

                # COL: GT rojo, PRED verde
                for lane_idx in range(labels_col.shape[1]):
                    xs, ys = [], []
                    for c, ca in enumerate(col_anchor):
                        row_id = labels_col[c, lane_idx]
                        if row_id != -1:
                            x = int(ca * W_img)
                            y = int(row_id / cfg.num_cell_row * H_img)
                            xs.append(x); ys.append(y)
                    if len(xs) > 1:
                        plt.plot(xs, ys, 'r.-', markersize=2, linewidth=1)

                for lane_idx in range(pred_col_np.shape[0]):
                    xs, ys = [], []
                    for c, ca in enumerate(col_anchor):
                        y = int(pred_col_np[lane_idx, c] / cfg.num_cell_row * H_img)
                        x = int(ca * W_img)
                        xs.append(x); ys.append(y)
                    if len(xs) > 1:
                        plt.plot(xs, ys, 'g.-', markersize=2, linewidth=1)

                plt.axis("off")
                plt.title(f"GT (rojo) vs Pred (verde) - iter {step}")
                plt.show()


# =========================
#   TRAIN LOOP
# =========================
def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, dataset):
    net.train()
    progress_bar = dist_tqdm(data_loader)

    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx

        # Debug del primer batch
        if b_idx == 0:
            print("\n=== DEBUG DATA_LABEL (primer batch de epoch) ===")
            for k, v in data_label.items():
                if torch.is_tensor(v):
                    print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                          f"min={v.min().item()}, max={v.max().item()}")
                else:
                    print(f"{k}: type={type(v)}")

        results = inference(net, data_label, dataset)

        if b_idx == 0:
            print("\n=== DEBUG RESULTS (primer batch) ===")
            for k, v in results.items():
                if torch.is_tensor(v):
                    print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                          f"min={v.min().item()}, max={v.max().item()}")
                else:
                    print(f"{k}: type={type(v)}")
            if "cls_label_col" in results:
                print("\nSample cls_label_col[0]:", results["cls_label_col"][0])
            if "cls_label" in results:
                print("\nSample cls_label[0]:", results["cls_label"][0])
            vr_col = (results["cls_label_col"] != -1).float().mean().item()
            vr_row = (results["cls_label"] != -1).float().mean().item()
            print(f"\nValid ratio col: {vr_col:.3f}, row: {vr_row:.3f}")

        loss = calc_loss(loss_dict, results, logger, global_step, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name,
                                  me_op.get(), global_step=global_step)
            logger.add_scalar(
                'meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {k: v for k, v in kwargs.items() if 'lane' not in k}
                progress_bar.set_postfix(loss='%.3f' % float(loss), **new_kwargs)


# =========================
#   MAIN
# =========================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    # Directorio de trabajo / distributed (simple)
    work_dir = get_work_dir(cfg) if args.local_rank == 0 else None
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101', '152',
                            '50next', '101next', '50wide', '101wide', '34fca']

    # Loaders / modelo / optim & sched / logger
    train_loader = get_train_loader(cfg)
    net = get_model(cfg)

    optimizer = get_optimizer(net, cfg)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)

    # =========================
    #    DEBUG OVERFIT (ON/OFF)
    # =========================
    DEBUG_OVERFIT = False  # ← pon a False para entrenamiento normal
    if DEBUG_OVERFIT:
        overfit_one_batch(net, train_loader, cfg, loss_dict, metric_dict, logger, iters=200, plot_pred=True)
        exit()  # termina tras el debug (cámbialo a False para seguir con el entrenamiento normal)

    # =========================
    #    EVAL FIRST (ON/OFF)
    # =========================
    EVAL_FIRST = False   # ← pon True si quieres correr eval_lane antes de entrenar
    if EVAL_FIRST:
        print("\n[DEBUG] Ejecutando evaluación inicial sin entrenamiento...\n")
        res = eval_lane(net, cfg, ep=0, logger=logger)
        print(f"Resultado evaluación inicial: {res}")
        
    # ===== ENTRENAMIENTO NORMAL =====
    max_res = 0
    res = None
    resume_epoch = 0
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1

            # Debug: lista de capas cargadas
        print("=== Pesos cargados desde checkpoint ===")
        print(f"Archivo: {cfg.resume}")
        print("Claves en state_dict:")
        print(list(resume_dict['model'].keys())[:10], "...")  # imprime las primeras 10
        print(f"Total de capas: {len(resume_dict['model'])}")

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)

import os, glob, shutil, torch

KEEP_LAST = 50     # cuántos checkpoints mantener
SAVE_EVERY = 1     # guarda cada N épocas
DO_EVAL = False    # <- desactiva evaluación

os.makedirs(work_dir, exist_ok=True)
max_res = -1.0

for epoch in range(resume_epoch, cfg.epoch):
    # Si tu función train devuelve pérdida media, captúrala; si no, deja None
    avg_loss = train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.dataset)
    train_loader.reset()

    if DO_EVAL:
        res = eval_lane(net, cfg, ep=epoch, logger=logger)
        if res is not None and res > max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)
        logger.add_scalar('CuEval/X', max_res, global_step=epoch)
    else:
        # Guarda cada N épocas (p.ej., N=1) SOBREESCRIBIENDO SIEMPRE "last.pth"
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(work_dir, 'last.pth')
            state = {
                'epoch': epoch,
                'model': (net.module if hasattr(net, 'module') else net).state_dict(),
                'optimizer': optimizer.state_dict(),
                # ← 1 línea “a prueba de fallo” por si el scheduler no tiene .state_dict()
                'scheduler': (scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, "state_dict") and callable(getattr(scheduler, "state_dict"))) else None),
                'cfg': getattr(cfg, '__dict__', cfg),
            }
            tmp_path = ckpt_path + '.tmp'            # escritura atómica
            torch.save(state, tmp_path)
            os.replace(tmp_path, ckpt_path)          # reemplaza el anterior

        # Loguea la métrica de entrenamiento
        if avg_loss is not None:
            logger.add_scalar('train/avg_loss', float(avg_loss), global_step=epoch)
        elif 'loss' in metric_dict:
            logger.add_scalar('train/loss', float(metric_dict['loss']), global_step=epoch)

logger.close()
