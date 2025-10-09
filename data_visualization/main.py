import os
import torch
from data_vis.data_load import load_m_rate_scence_class_rank,load_cutoff_data
from data_vis.draw import draw_box,draw_cutoff

def vis_discussion_1():
    '''不同变异率下所有场景的箱线图'''
    '''论文插图：不同变异率对FP class rank的影响'''
    data = load_m_rate_scence_class_rank()
    save_dir = "imgs"
    save_file_name = "discussion_1.png"
    save_path = os.path.join(save_dir, save_file_name)
    draw_box(data,save_path)

def vis_cutoff():
    '''论文插图，cutoff对PN/ASR/ACC的影响'''
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = 'WaNet'
    random_seed = 1
    device = torch.device("cuda:1")
    CutOff_list,PN_rate_list,ASR_list,ACC_list = load_cutoff_data(dataset_name,model_name,attack_name,random_seed, device)
    save_dir = os.path.join("imgs", "cutoff")
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_cutoff(CutOff_list,PN_rate_list,ASR_list,ACC_list, save_path)


if __name__ == "__main__":
    vis_cutoff()