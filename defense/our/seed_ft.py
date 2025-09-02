def seed_ft(model, filtered_poisoned_testset, poisoned_trainset, clean_testset, seedSet, device, poisoned_ids, logger):
    # FT前模型评估
    e = EvalModel(model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    print("backdoor_ASR:",asr)
    e = EvalModel(model,clean_testset,device)
    acc = e.eval_acc()
    print("backdoor_acc:",acc)
    poisoned_evalset_loader = DataLoader(
                poisoned_trainset, # 非预制
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    # 冻结
    freeze_model(model,dataset_name=dataset_name,model_name=model_name)
    # 获得class_rank
    class_rank = get_classes_rank_v2(exp_root_dir,dataset_name,model_name,attack_name)
    # 基于种子集和后门模型微调10轮次
    # BadNets:3,IAD:3,Refool:10,WaNet:20
    _, model = train(model,device,seedSet,num_epoch=20,lr=1e-3,logger=logger)
    e = EvalModel(model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    print("FT_ASR:",asr)
    e = EvalModel(model,clean_testset,device)
    acc = e.eval_acc()
    print("FT_acc:",acc)
    ranked_sample_id_list_1, isPoisoned_list_1, _ = sort_sample_id(model,device,
                poisoned_evalset_loader,
                poisoned_ids,
                class_rank=None)
    ranked_sample_id_list_2, isPoisoned_list_2, _ = sort_sample_id(model,device,
            poisoned_evalset_loader,
            poisoned_ids,
            class_rank=class_rank)
    # draw(isPoisoned_list_1,isPoisoned_list_2, file_name=f"{attack_name}.png")