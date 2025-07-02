import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

plt.rcParams['font.sans-serif'] = ['Times']
plt.rcParams['font.size'] = '20'
pred_PATH = './data/myf0/'
f0_path = 'data/'
name_list = ['opera_male3', 'opera_male5', 'pop2', 'pop3', 'train03', 'train04', 'train05', 'train06']

for name in name_list:
    model_type = 0
    save_name = name
    start = 0
    end = 1300
    if model_type == 1:
        save_name = name + '_unet'
        print('We are printing unet prediction!')
    else:
        print('We are printing the result of the proposed model!')
    pitch = np.loadtxt(f0_path + 'f0ref/' + name + '.txt')
    x = pitch[:, 0] * 100
    y_ground = pitch[:, 1]
    if model_type == 0:
        y_pred = np.loadtxt(pred_PATH + name + '.txt')[:, 1]
        print('We are loading the result of the proposed model!')
    if model_type == 1:
        y_pred = np.loadtxt(name + '.txt')[:, 1]
        print('We are loading unet prediction!')
    ax = plt.subplot(111)
    fontsize = 18
    ax.set_xlabel('Time: (ms)', fontsize=fontsize)
    ax.set_ylabel('Hz', fontsize=fontsize)
    axes = plt.gca()
    axes.set_ylim([0, 600])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    print(len(y_pred))
    ground = plt.plot(x[start:end], y_ground[start:end], label='Ground Truth', color='blue')
    predict = plt.plot(x[start:end], y_pred[start:end], label='Prediction', color='green')
    plt.legend(loc='upper left', fontsize=20)

    plt.subplots_adjust(top=0.95, bottom=0.14, left=0.133, right=0.93, hspace=0, wspace=0)
    plt.savefig('./data/pic/' + save_name + '.png', format='png', dpi=640)
    plt.clf()

print('done')
