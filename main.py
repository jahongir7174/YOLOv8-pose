import argparse
import copy
import csv
import os
import warnings

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    # Model
    model = nn.yolo_v8_n(len(params['names']))
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open('../Dataset/COCOPose/train2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../Dataset/COCOPose/images/train2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, True)

    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'BoxAP', 'PoseAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()

            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()

            for i, (samples, targets) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', m_loss.avg)
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'BoxAP': str(f'{last[0]:.3f}'),
                                 'PoseAP': str(f'{last[1]:.3f}')})
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                ckpt = {'model': copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(ckpt, './weights/last.pt')
                if best == last[1]:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    with open('../Dataset/COCOPose/val2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../Dataset/COCOPose/images/val2017/' + filename)
    numpy.random.shuffle(filenames)
    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 4, False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    box_mean_ap = 0.
    kpt_mean_ap = 0.
    box_metrics = []
    kpt_metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 2) % ('BoxAP', 'PoseAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.head.nc)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]
            kpt = targets['kpt'][idx]

            cls = cls.cuda()
            box = box.cuda()
            kpt = kpt.cuda()

            correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()  # init
            correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()  # init

            if output.shape[0] == 0:
                if cls.shape[0]:
                    box_metrics.append((correct_box,
                                        *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    kpt_metrics.append((correct_kpt,
                                        *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue

            # Predictions
            pred = output.clone()
            p_kpt = pred[:, 6:].view(output.shape[0], kpt.shape[1], -1)

            # Evaluate
            if cls.shape[0]:
                t_box = util.wh2xy(box)
                t_kpt = kpt.clone()
                t_kpt[..., 0] *= w
                t_kpt[..., 1] *= h

                target = torch.cat((cls, t_box * scale), 1)  # native-space labels
                correct_box = util.compute_metric(pred[:, :6], target, iou_v)
                correct_kpt = util.compute_metric(pred[:, :6], target, iou_v, p_kpt, t_kpt)
            # Append
            box_metrics.append((correct_box, output[:, 4], output[:, 5], cls.squeeze(-1)))
            kpt_metrics.append((correct_kpt, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    box_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*box_metrics)]  # to numpy
    kpt_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*kpt_metrics)]  # to numpy
    if len(box_metrics) and box_metrics[0].any():
        tp, fp, m_pre, m_rec, map50, box_mean_ap = util.compute_ap(*box_metrics)
    if len(kpt_metrics) and kpt_metrics[0].any():
        tp, fp, m_pre, m_rec, map50, kpt_mean_ap = util.compute_ap(*kpt_metrics)
    # Print results
    print('%10.3g' * 2 % (box_mean_ap, kpt_mean_ap))

    # Return results
    model.float()  # for training
    return box_mean_ap, kpt_mean_ap


@torch.no_grad()
def demo(args):
    import cv2
    palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    stride = int(max(model.stride.cpu().numpy()))

    model.half()
    model.eval()

    camera = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while camera.isOpened():
        # Capture frame-by-frame
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]  # current shape [height, width]
            r = min(1.0, args.input_size / shape[0], args.input_size / shape[1])
            pad = int(round(shape[1] * r)), int(round(shape[0] * r))
            w = args.input_size - pad[0]
            h = args.input_size - pad[1]
            w = numpy.mod(w, stride)
            h = numpy.mod(h, stride)
            w /= 2
            h /= 2
            if shape[::-1] != pad:  # resize
                image = cv2.resize(image,
                                   dsize=pad,
                                   interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image,
                                       top, bottom,
                                       left, right,
                                       cv2.BORDER_CONSTANT)  # add border
            # Convert HWC to CHW, BGR to RGB
            image = image.transpose((2, 0, 1))[::-1]
            image = numpy.ascontiguousarray(image)
            image = torch.from_numpy(image)
            image = image.unsqueeze(dim=0)
            image = image.cuda()
            image = image.half()
            image = image / 255
            # Inference
            outputs = model(image)
            # NMS
            outputs = util.non_max_suppression(outputs, 0.25, 0.7, model.head.nc)
            for output in outputs:
                output = output.clone()
                if len(output):
                    box_output = output[:, :6]
                    kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
                else:
                    box_output = output[:, :6]
                    kps_output = output[:, 6:]

                r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

                box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                box_output[:, :4] /= r

                box_output[:, 0].clamp_(0, shape[1])  # x
                box_output[:, 1].clamp_(0, shape[0])  # y
                box_output[:, 2].clamp_(0, shape[1])  # x
                box_output[:, 3].clamp_(0, shape[0])  # y

                kps_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                kps_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                kps_output[..., 0] /= r
                kps_output[..., 1] /= r
                kps_output[..., 0].clamp_(0, shape[1])  # x
                kps_output[..., 1].clamp_(0, shape[0])  # y

                for box in box_output:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    cv2.rectangle(frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                for kpt in reversed(kps_output):
                    for i, k in enumerate(kpt):
                        color_k = [int(x) for x in kpt_color[i]]
                        x_coord, y_coord = k[0], k[1]
                        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                            if len(k) == 3:
                                conf = k[2]
                                if conf < 0.5:
                                    continue
                            cv2.circle(frame,
                                       (int(x_coord), int(y_coord)),
                                       5, color_k, -1, lineType=cv2.LINE_AA)
                    for i, sk in enumerate(skeleton):
                        pos1 = (int(kpt[(sk[0] - 1), 0]), int(kpt[(sk[0] - 1), 1]))
                        pos2 = (int(kpt[(sk[1] - 1), 0]), int(kpt[(sk[1] - 1), 1]))
                        if kpt.shape[-1] == 3:
                            conf1 = kpt[(sk[0] - 1), 2]
                            conf2 = kpt[(sk[1] - 1), 2]
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue
                        if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                            continue
                        if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                            continue
                        cv2.line(frame,
                                 pos1, pos2,
                                 [int(x) for x in limb_color[i]],
                                 thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    camera.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def profile(args, params):
    model = nn.yolo_v8_n(len(params['names']))
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))
    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    profile(args, params)
    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
