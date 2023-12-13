import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def draw_box(data,labels,title,save_path):
    # rcParams['font.size'] = 8   # 设置字体大小为14
    fig = plt.figure()
    plt.boxplot(
            x=data, # 需要绘制的数据
            vert=True, # 垂直排列箱线图
            widths=0.3, # 箱形宽度
            labels=labels,      # 箱形图的标签
            patch_artist=True,  # 是否为箱子填充颜色，默认为False
            medianprops={       # 设置中位线属性
                'linestyle': '-', 'color': 'r', 'linewidth': 1.5
            },
            showmeans=True,     # 是否显示均值点，默认为False
            meanline=True,      # 是否显示均值线，默认为False
            meanprops={         # 设置均值点属性
                'marker': 'o', 'markersize': 7.5, 'markeredgewidth': 0.75, 'markerfacecolor': '#b7e1a1', 'markeredgecolor': 'r', 'color': 'k', 'linewidth': 1.5
            },
            showfliers=True,    # 是否显示异常值，默认为True
            flierprops={        # 设置异常点属性
                'marker': '^', 'markersize': 6.75, 'markeredgewidth': 0.75, 'markerfacecolor': '#ee5500', 'markeredgecolor': 'k'
            },
            whiskerprops={      # 设置须的线条属性
                'linestyle': '--', 'linewidth': 1.2, 'color': '#480656'
            },
            capprops={
                'linestyle': '-', 'linewidth': 1.5, 'color': '#480656'
            }
            
        )
    
    plt.title(title, fontsize=10)
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig(save_path,transparent=False,dpi=600)
    
if __name__ == "__main__":

    all_data = [np.random.normal(0, std, size=100) for std in range(1,4)]
    labels = ['$x_1$', '$x_2$', '$x_3$']
    title =  "box"
    save_path = "temp.png"
    draw_box(all_data, labels, title, save_path)