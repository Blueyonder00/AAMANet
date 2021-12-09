from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

# opts['init_model_path'] = None
# opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
# opts['init_model_path'] = '../models/MANet_105.pth'
#opts['init_model_path'] = '/home/pjc/szh/MANet/models/MANet-A-708-2253574.pth'

opts['model_path'] = '../models/MANet-A-717-2313-'
opts['rgb_log_dir'] = '../models/MANet-A-713-2313/log'
opts['batch_frames'] = 8  #8
opts['batch_pos'] = 32 #32
opts['batch_neg'] = 96  #96

opts['overlap_pos'] = [0.7, 1]  #[0.7,1] todo
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding'] = 16

opts['lr'] = 0.00012
opts['w_decay'] =0.0005   #0.0005
opts['momentum'] = 0.9
opts['clip_value'] = 10
opts['grad_clip'] = 10
opts['ft_layers'] = ['fc', 'R', 'T', 'conv', 'w']
opts['lr_mult'] = {'fc': 2,'w': 2, 'R': 1, 'T': 1, 'conv': 1}

opts['n_cycles'] = 300

opts['log_dir'] = '../log'
