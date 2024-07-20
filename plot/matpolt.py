import matplotlib.pyplot as plt

# 读取数据
# file_path = 'results/cifar10/test_local_epoch_test_loss_acc.txt'
# file_path = "results/cifar10/test_splitpoint_2_local_epoch_test_loss_acc.txt"
# file_path = "results/cifar10/train_splitpoint_3_loss_acc.txt"
file_path = "results/cifar10/train_splitpoint_1_loss_acc.txt"
epochs = []
accuracy = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    # for line in lines[304:652]:
    for line in lines:
        parts = line.strip().split()
        epochs.append(int(parts[0]))
        accuracy.append(float(parts[3]))


plt.figure()
plt.plot(epochs, accuracy)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy')
# plt.yticks([0.3,0.5,0.7, 0.9,0.95, 1.0])
plt.ylim(0.3, 1.0) 
plt.xticks(range(0, max(epochs)+1, 100))

# plt.axhline(y=0.5, color='black', linewidth=1)
# plt.axhline(y=0, color='white', linewidth=0)  # 将原点的线隐藏

plt.show()

# 保存图表
plt.savefig('plot/figure/train_splitpoint_1_loss_acc.pdf')
