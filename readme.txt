1、首先我们需要先运行cut.py文件把训练数据进行切分储存起来
python cut.py --input_folder 数据路径  --output_folder 切分后储存路径 --patch_size 切分大小32或者64
2、train_new.py 训练代码
python train_new.py --train_path 数据路径 --result_pth 结果保存 --pathsize 32或者64,要与切割的对应
3、test_bigimg.py 测试代码
python test_bigimg.py --weights_file 权重文件路径 --patchsize 32或者64 --input_dir 测试数据路径  --output_dir 结果保存路径

其中patchsize训练和测试的时候大小要对应

**Download links for the Drone Vehicle and AVIID datasets**
