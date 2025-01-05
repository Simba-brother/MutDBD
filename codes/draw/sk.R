# Import libraries
library(ScottKnottESD)
library(readr)
library(ggplot2)

# load data
box_csv_path = "/data/mml/backdoor_detect/experiments/SK/CIFAR10/ResNet18/BadNets/0.01/box_precision.csv"
model_performance <- read_csv(box_csv_path)

# apply ScottKnottESD and prepare a ScottKnottESD dataframe
sk_results <- sk_esd(model_performance)

sk_ranks <- data.frame(
  class = names(sk_results$groups),
  rank = paste0("Rank-", sk_results$groups)
)

# prepare a dataframe for generating a visualisation

plot_data <- melt(model_performance)
# print(colnames(plot_data))
plot_data <- merge(plot_data, sk_ranks, by.x = "variable", by.y = "class")


# generate a visualisation
g <- ggplot(data = plot_data, aes(x = variable, y = value, fill = rank)) +
  geom_boxplot() +
  stat_summary(fun="mean", geom="point", shape=20, size=2.5, color="red", fill="red",alpha=0.7) +
  ylim(c(0, 1)) +
  facet_grid(~rank, scales = "free_x") +
  scale_fill_brewer(direction = -1) +
  ylab("precision") + xlab("Class") + ggtitle("") + theme_bw() +
  theme(text = element_text(size = 12),axis.text.x = element_text(size=8,angle = 90),
  legend.position = "none")
ggsave("myplot.png",g)