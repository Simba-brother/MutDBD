
import matplotlib.pyplot as plt
def draw_line(x_ticks, title:str, xlabel:str, ylabel:str, save_path:str, draw_data_dict:dict):
    '''
    x_ticks:list
        x轴的刻度
    title:str
        图像标题
    xlabel:str
        x轴的含义
    ylabel:str
        y轴的含义
    save_path:str
        图像保存路径
    draw_data_dict:dict
        {str:[]},一个key一条折线
    '''
    # 设置图片大小，清晰度
    # plt.figure(figsize=(20, 8), dpi=800)
    x_list = [x for x in list(range(len(x_ticks)))]
    for key,value in draw_data_dict.items():
        plt.plot(x_list, value, label=key, marker='o')
    # 设置x轴的刻度
    font_size=10
    plt.xticks(x_list,x_ticks,fontsize=font_size) # rotation=45
    plt.xlabel(xlabel,fontsize=font_size)
    plt.ylabel(ylabel,fontsize=font_size)
    plt.title(title,fontsize=font_size)
    plt.tight_layout(pad=0)
    # 绘制网格(控制透明度)
    plt.grid(alpha=0.5, linestyle=':')
    # 添加图例
    plt.legend()
    plt.show()
    plt.savefig(save_path,transparent=False,dpi=600)