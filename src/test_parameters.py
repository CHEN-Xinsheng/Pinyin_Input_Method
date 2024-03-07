import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D	
from tqdm import tqdm
from utils import ROOT_DIR



def model_2(training_set: str):
	
	# 使用的模型
	model = 2

	# 定义 x 和 y 的取值范围
	x = np.linspace(0, 1, 11)

	word_accuracy = np.empty(len(x))
	sentence_accuracy = np.empty(len(x))

	for i in tqdm(range(len(x))):
		# 设置参数
		cmd = ['python', ROOT_DIR / 'src' / 'pinyin.py', '-s', f'{training_set}', '-m', f'{model}', '-c', str(x[i]), '0']
		# 运行 pinyin.py 并获取输出
		output = subprocess.check_output(cmd, universal_newlines=False).decode('utf-8')
		# 从输出中捕获字准确率和句准确率
		word_accuracy[i] = float(re.search(r'字准确率：(\d+\.\d+)', output).group(1))
		sentence_accuracy[i] = float(re.search(r'句准确率：(\d+\.\d+)', output).group(1))

	# 将测试结果写入文件
	with open(ROOT_DIR / 'parameters' / f'{training_set}_2_rough.txt', 'w') as f:
		for i in range(len(x)):
			f.write('x = %.4f, word_acc = %.4f, sentence_acc = %.4f' % (x[i], word_accuracy[i], sentence_accuracy[i]))

	# 绘制图像
	plt.plot(x, word_accuracy)

	# 添加标题和坐标轴标签
	plt.title('word_accuracy vs x')
	plt.xlabel('x')
	plt.ylabel('word_accuracy')
	plt.savefig(ROOT_DIR / 'parameters' / f'word-acc_{training_set}_2_rough.png')
	plt.clf()

	# 绘制图像
	plt.plot(x, sentence_accuracy)

	# 添加标题和坐标轴标签
	plt.title('sentence_accuracy vs x')
	plt.xlabel('x')
	plt.ylabel('sentence_accuracy')
	plt.savefig(ROOT_DIR / 'parameters' / f'sentence-acc_{training_set}_2_rough.png')


if __name__ == '__main__':

	model_2('sina')

