# MNIST_Early_Stopping.py
这是关于early_stop，首先利用自定义early_stopping工具，（把valid 中的-loss作为分值标准，记录最好分值），如果，继续训练发现越来越差，则经设定好的阈值（训练几个还不见好）就停止训练，而不是一直训练完设定的epoch次数）
这是有关early_stop，即既然记录最好的训练模型参数及状态，也能避免不必要的训练（即越来越差，就没必要训练完所有epoch次数）。early_stop是以 valid中的loss为依据的，其loss最好时则是 整个模型最好的参数。early_stop里面，初始时以valid_loss 中的 -valid_loss为分值，保存状态； 若valid_loss每继续降低，就会保存状态；若分值还没之前好，则设定一个参数和阈值，允许其最多能经历 该阈值次的这样糟糕的训练，每次训练比之前的差，该参数会+=1， 直至该参数>=阈值， 则无法忍受原来越糟糕的训练，提前结束训练。
1. MNIST_Early_Stopping.py中
def create_datasets(batch_size):
   return train_loader, test_loader, valid_loader
   
def train_model(model, batch_size, patience, n_epoches):
   return model, avg_train_losses, avg_valid_losses
    # avg_train_losses, avg_valid_losses为epoch个loss 的数组即 [...epoch...个的loss数] 用来描点画图， 并在图中标出early_stop的点，纵坐标是             
    # valid_loss， 横坐标是 range(1, len(valid_loss) + 1， 所以valid_loss.index(min(valid_loss)) + 1 即为提前停止点。
class Net(nn.Module):   # 定义全连接型的网络
   return x  
   
2. pytorch.py中 
class EarlyStopping:  # 中有三个方法
   def __init__(self, patience=7, verbose=False):
   
   def __call__(self, val_loss, model):       # 以 -valid_loss为分值，判断valid_loss训练后的分值比之前的怎么样
   
   def save_checkpoint(self, val_loss, model): # 里面打印loss更新情况，保存模型参数状态
  
3. class_correct = list(0. for i in range(10))   # list(0.0, 0.0, ....10个0.0)
