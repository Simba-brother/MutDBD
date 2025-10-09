import time
import torch
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from defense.asd.core.log import Record,AverageMeter,tabulate_step_meter,tabulate_epoch_meter
from itertools import cycle,islice

def linear_test(model, loader, criterion, device,logger):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [loss_meter, acc_meter]

    model.eval()
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        # data = batch["img"]
        # target = batch["target"]
        data = batch[0]
        target = batch[1]
        data = data.to(device)
        target = target.to(device)
        # data = batch["img"].cuda(gpu, non_blocking=True)
        # target = batch["target"].cuda(gpu, non_blocking=True)
        with torch.no_grad():
            output = model(data)
        criterion.reduction = "mean"
        loss = criterion(output, target)

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((torch.sum(truth).float() / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 2, meter_list,logger)

    logger.info("Linear test summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list,logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result

def poison_linear_record(model, loader, criterion, device, **kwargs):
    # 数据集数量
    num_data = len(loader.dataset)
    # target_record = Record("target", num_data) # 记录标签 
    # poison_record = Record("poison", num_data) # 是否被污染
    # origin_record = Record("origin", num_data) # 原本的target
    loss_record = Record("loss", num_data) # 记录每个样本的loss
    '''
    dataset_name = kwargs["dataset_name"]
    model_name = kwargs["model_name"]
    
    if dataset_name in ["CIFAR10","GTSRB"]:
        if model_name == "ResNet18":
            in_features = model.classifier.in_features
            node_str = "linear"
        elif model_name == "VGG19":
            in_features = model.classifier_2.in_features
            node_str = "classifier"
        elif model_name == "DenseNet":
            in_features = model.classifier.in_features
            node_str = "linear"
    elif dataset_name == "ImageNet2012_subset":
        if model_name == "ResNet18":
            in_features = model.fc.in_features
            node_str = "flatten"
        elif model_name == "VGG19":
            in_features = model.classifier[-1].in_features
            node_str = "classifier.5"
        elif model_name == "DenseNet":
            in_features = model.classifier.in_features
            node_str = "flatten"

    feature_record = Record("feature", (num_data, in_features))
    '''
    # feature_record = Record("feature", (num_data, model.backbone.feature_dim))
    record_list = [
        # target_record,
        # poison_record,
        # origin_record,
        loss_record
        # feature_record,
    ]

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(loader): # 分批次遍历数据加载器
            # data = batch["img"].to(device)
            # target = batch["target"].to(device)
            data = batch[0].to(device)
            target = batch[1].to(device)
            with torch.no_grad():
                # feature_extractor = create_feature_extractor(model, return_nodes=[node_str])
                # feature_dic = feature_extractor(data)
                # feature = feature_dic[node_str]
                output = model(data)
            criterion.reduction = "none" # 数据不进行规约,以此来得到每个样本的loss,而不是批次的avg_loss
            raw_loss = criterion(output, target)

            # target_record.update(target)
            # poison_record.update(batch["poison"])
            # origin_record.update(batch["origin"])
            loss_record.update(raw_loss.cpu())
            # feature_record.update(feature.cpu())

    return record_list

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]

def mixmatch_train(model, xloader, uloader, criterion, optimizer, epoch, device,logger, **kwargs):
    '''
    loss_meter = AverageMeter("loss")
    xloss_meter = AverageMeter("xloss")
    uloss_meter = AverageMeter("uloss")
    lambda_u_meter = AverageMeter("lambda_u")
    meter_list = [loss_meter, xloss_meter, uloss_meter, lambda_u_meter]
    '''

    # 数据加载器转化成迭代器
    xiter = cycle(xloader) # 有监督
    uiter = cycle(uloader) # 无监督
    xlimited_cycled_data = islice(xiter,0,kwargs["train_iteration"])
    ulimited_cycled_data = islice(uiter,0,kwargs["train_iteration"])
    model.train()
    '''
    loss_list = []
    xloss_list = []
    uloss_list = []
    lambda_list = []
    '''
    # start = time.time()
    
    '''
    for batch_idx in range(kwargs["train_iteration"]):
        # batch_start_time = time.perf_counter()

        # load_batch_data_start_time = time.perf_counter()
         # 加载batch data
        xbatch = next(xiter)
        xinput, xtarget = xbatch["img"], xbatch["target"]
    
        ubatch = next(uiter)
        uinput1, uinput2 = ubatch["img1"], ubatch["img2"]

        # load_batch_data_end_time = time.perf_counter()
        # load_batch_data_cost_time = load_batch_data_end_time - load_batch_data_start_time
        # hours, minutes, seconds = convert_to_hms(load_batch_data_cost_time) 
        # logger.info(f"加载一批数据耗时:{hours}时{minutes}分{seconds:.3f}秒")
    '''
    for batch_idx,(xbatch,ubatch) in enumerate(zip(xlimited_cycled_data,ulimited_cycled_data)):
        xinput, xtarget = xbatch["img"], xbatch["target"]
        uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        '''
        try:
            xbatch = next(xiter) # 带标签中的一个批次
            xinput, xtarget = xbatch["img"], xbatch["target"]
        except:
            # 如果迭代器走到最后无了,则从头再来迭代
            # iter_xbatch_start_time = time.perf_counter()
            xiter = iter(xloader)
            xbatch = next(xiter)
            xinput, xtarget = xbatch["img"], xbatch["target"]
            # iter_xbatch_end_time = time.perf_counter()
            # iter_xbatch_cost_time = iter_xbatch_end_time - iter_xbatch_start_time
            # hours, minutes, seconds = convert_to_hms(iter_xbatch_cost_time)
            # print(f"iter_xbatch耗时:{hours}时{minutes}分{seconds:.3f}秒")
        try:
            # 无标签batch
            ubatch = next(uiter) # 不带标签中的一个批次
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        except:
            # 如果迭代器走到最后无了,则从头再来迭代
            # iter_ubatch_start_time = time.perf_counter()
            uiter = iter(uloader)
            ubatch = next(uiter)
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
            # iter_ubatch_end_time = time.perf_counter()
            # iter_ubatch_cost_time = iter_ubatch_end_time - iter_ubatch_start_time
            # hours, minutes, seconds = convert_to_hms(iter_ubatch_cost_time)
            # print(f"iter_ubatch耗时:{hours}时{minutes}分{seconds:.3f}秒")'
        '''
        # before_calcu_loss_start_time = time.perf_counter()
        batch_size = xinput.size(0)
        xtarget = torch.zeros(batch_size, kwargs["num_classes"]).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )

        # batch数据放到gpu上
        xinput = xinput.to(device)
        xtarget = xtarget.to(device) 
        uinput1 = uinput1.to(device)
        uinput2 = uinput2.to(device)
        # uinput2 = uinput2.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / kwargs["temperature"])
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()


        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(kwargs["alpha"], kwargs["alpha"])
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)
        # before_calcu_loss_end_time = time.perf_counter()
        # before_calcu_loss_cost_time = before_calcu_loss_end_time - before_calcu_loss_start_time
        # hours, minutes, seconds = convert_to_hms(before_calcu_loss_cost_time) 
        # logger.info(f"计算损失之前的准备工作耗时:{hours}时{minutes}分{seconds:.3f}秒")

        # 计算损失
        # calcu_loss_start_time = time.perf_counter()
        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / kwargs["train_iteration"],
        )
        loss = Lx + lambda_u * Lu
        # calcu_loss_end_time = time.perf_counter()
        # calcu_loss_cost_time = calcu_loss_end_time - calcu_loss_start_time 
        # hours, minutes, seconds = convert_to_hms(calcu_loss_cost_time)
        # logger.info(f"计算损失耗时:{hours}时{minutes}分{seconds:.3f}秒")

        # BP_start_time = time.perf_counter()
        optimizer.zero_grad() # 优化器梯度清0
        loss.backward() # 基于loss函数求梯度值
        optimizer.step() # 优化器优化基于损失函数求的梯度优化模型参数
        # BP_end_time = time.perf_counter()
        # BP_cost_time = BP_end_time - BP_start_time
        # hours, minutes, seconds = convert_to_hms(BP_cost_time)
        # logger.info(f"反向传播耗时:{hours}时{minutes}分{seconds:.3f}秒")
        '''
        loss_list.append(loss.item())
        xloss_list.append(Lx.item())
        uloss_list.append(Lu.item())
        lambda_list.append(lambda_u)
        '''
        '''
        write_start_time = time.perf_counter()
        loss_meter.update(loss.item())
        xloss_meter.update(Lx.item())
        uloss_meter.update(Lu.item())
        lambda_u_meter.update(lambda_u)
        tabulate_step_meter(batch_idx, kwargs["train_iteration"], 3, meter_list,logger)
        write_end_time = time.perf_counter()
        write_cost_time = write_end_time - write_start_time
        hours, minutes, seconds = convert_to_hms(write_cost_time)
        logger.info(f"制表写结果耗时:{hours}时{minutes}分{seconds:.3f}秒")
        '''
        '''
        batch_end_time = time.perf_counter()
        batch_cost_time = batch_end_time -batch_start_time
        hours, minutes, seconds = convert_to_hms(batch_cost_time)
        logger.info(f"batch总耗时:{hours}时{minutes}分{seconds:.3f}秒")
        '''
    '''
    avg_loss = round(sum(loss_list) / len(loss_list),4)
    avg_xloss =round(sum(xloss_list) / len(xloss_list),4)
    avg_uloss = round(sum(uloss_list) / len(uloss_list),4)
    avg_lambda = round(sum(lambda_list) / len(lambda_list),4)
    logger.info(f"Epoch:{epoch}, avg_loss:{avg_loss}, avg_xloss:{avg_xloss}, avg_uloss:{avg_uloss}, avg_lambda:{avg_lambda}")
    '''
    '''
    logger.info("MixMatch training summary:")
    tabulate_epoch_meter(time.time() - start, meter_list,logger)
    result = {m.name: m.total_avg for m in meter_list}
    return result
    '''
