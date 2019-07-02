import os
import time
import cv2
import face_model
import numpy as np
import pickle


class Args():
	def __init__(self):
		self.image_size = '112,112'
		self.gpu = 0
		self.model = ''
		self.ga_model = ''
		self.threshold = 1.24
		self.flip = 0
		self.det = 0


def mkface(imgs_dir, faces_save_dst):
	args = Args()
	model = face_model.FaceModel(args)

	if not os.path.exists(faces_save_dst):
		os.mkdir(faces_save_dst)

	folders = os.listdir(imgs_dir)

	for folder in folders:
		imgs = os.listdir(os.path.join(imgs_dir, folder))
		n = len(imgs)
		cnt = 0
		for img in imgs:
			start = time.time()
			img_root_path = os.path.join(imgs_dir, folder, img)
			try:
				pic = cv2.imread(img_root_path)
				pic = model.get_input(pic)  # 3 * 112 * 112
				if type(pic) == np.ndarray:
					if not os.path.exists(os.path.join(faces_save_dst, folder)):
						os.mkdir(os.path.join(faces_save_dst, folder))
					cv2.imwrite(os.path.join(faces_save_dst, folder, folder + str(cnt)+'.jpg'), np.transpose(pic, (1,2,0))[:,:,::-1])
					cnt += 1
				end = time.time()

				interval = end - start
			except:
				continue


def mklst(im2rec_path='/home/bsoft/insightface/deploy', lst_path='/home/bsoft/insightface/', lst_name='test_face', image_path='/home/bsoft/insightface/deploy/test_face'):
	os.system(f'python {im2rec_path}/face2rec.py --list --recursive {lst_path}/{lst_name} {image_path} ')
	os.system(f'python {im2rec_path}/face2rec.py {lst_path}/{lst_name}  {image_path}')


# 创建验证数据集,.bin文件
def mktxt(img_path, txt_path, seed=1234):
	np.random.seed(seed)

	with open(txt_path, 'w') as f:
		dic = {}

		# 相同人脸
		for person in os.listdir(img_path):
			dic[person] = []
			# current_img_path 是主路径下第一个图片文件夹
			current_img_path = os.path.join(img_path, person)
			if len(os.listdir(current_img_path)) <= 1:
				continue
			lenth = len(os.listdir(current_img_path))
			# idx0 从0开始
			for idx0, img_file in enumerate(os.listdir(current_img_path)):
				img0_path = os.path.join(current_img_path, img_file)
				dic[person].append(img0_path)

				if idx0 + 1 == lenth:
					break

				# 对剩余的idx进行循环
				for idx1 in range(idx0 + 1, lenth, 1):
					img1_path = os.path.join(current_img_path, os.listdir(current_img_path)[idx1])
					# 生成同一个人的两张图片的line
					line_same = img0_path + ',' + img1_path + ',' + '1'
					f.write(line_same)
					f.write('\n')
		# 针对不同人脸的操作
		len_dic = len(dic)
		for key0, value0 in dic.items():
			for img0_path in value0:
				for key1, value1 in dic.items():
					if key0 == key1:
						continue

					for img1_path in value1:
						line_diff = img0_path + ',' + img1_path + ',' + '0'
						f.write(line_diff)
						f.write('\n')


def mkbin(bin_path, txt_path):
	image_size = [112,112]
	pairs, same_pairs_counts = read_pairs(txt_path)
	paths, issame_list = get_paths(pairs=pairs, same_pairs=same_pairs_counts)
	lfw_bins = []
	i = 0
	for path in paths:
		# path_list = [(path0, path1),...]
		with open(path, 'rb') as fin:
			_bin = fin.read()
			lfw_bins.append(_bin)
			i += 1
			if i % 1000 == 0:
				print('loading pairs', i)

	with open(bin_path, 'wb') as f:
		pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)


def get_paths(pairs, same_pairs):
	nrof_skipped_pairs = 0
	path_list = []
	issame_list = []
	cnt = 1
	for pair in pairs:
		path0 = pair[0]
		path1 = pair[1]

		if cnt < same_pairs:
			issame = True
		else:
			issame = False
		if os.path.exists(path0) and os.path.exists(path1):
			path_list += (path0, path1)
			issame_list.append(issame)
		else:
			print('not exists', path0, path1)
			nrof_skipped_pairs += 1
		cnt += 1
	if nrof_skipped_pairs > 0:
		print(f'Skipped {nrof_skipped_pairs} image pairs')
	return path_list, issame_list


def read_pairs(pairs_filename):
	pairs = []
	same_pairs_counts = 0
	with open(pairs_filename, 'r') as f:
		for line in f.readlines()[0:]:
			pair = line.strip().split(',')
			if pair[2] == 1:
				same_pairs_counts += 1
			pairs.append(pair)
	return np.array(pairs), same_pairs_counts


def main():
	imgs_dir = '/home/bsoft/insightface/deploy/test_image'
	faces_save_dst = '/home/bsoft/insightface/deploy/test_face'
	im2rec_path = '/home/bsoft/insightface/deploy'
	lst_path = '/home/bsoft/insightface/'
	lst_name = 'test_face'
	txt_path = './face_val.txt'
	bin_path = './'

	# # make face data
	# mkface(imgs_dir=imgs_dir, faces_save_dst=faces_save_dst)
	# # make .lst and .rec
	# mklst(im2rec_path=im2rec_path, lst_path=lst_path, lst_name=lst_name)
	# # make txt file
	# mktxt(img_path=imgs_dir, txt_path=txt_path)
	# make .bin file
	mkbin(bin_path=bin_path, txt_path=txt_path)

if __name__ == '__main__':
	main()
