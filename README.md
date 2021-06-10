# PaddleNLP-Learning

## day02 结果复现

如果要复现day02 的结果，首先需要:

### 1. 准备好数据集

- 下载数据集
- 
wget https://dataset-bj.cdn.bcebos.com/qianyan/bq_corpus.zip

wget https://dataset-bj.cdn.bcebos.com/qianyan/lcqmc.zip

wget https://dataset-bj.cdn.bcebos.com/qianyan/paws-x-zh.zip

- 解压数据集到 ./data 目录
- 
unzip ./bq_corpus.zip -d ./data/

unzip ./lcqmc.zip -d ./data/

unzip ./paws-x-zh.zip -d ./data/

- 删除压缩包
- 
rm bq_corpus.zip lcqmc.zip paws-x-zh.zip

- 重新命名
- 
mv ./data/paws-x-zh/ ./data/paws-x/

### 2. 训练

python day02_mian.py

注: 直接运行运行 day02_mian.py，会在lcqmc, bq_corpus, paws-x 三个数据集分别训练三个模型。如果你只想在其中一个数据集上 Fine-tuning，到代码里去改把！
