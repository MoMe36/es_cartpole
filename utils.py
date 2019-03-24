from tensorboardX import SummaryWriter 
import os 
import shutil

def make_writer(path = './runs/CartPole/'):

    try: 
        os.makedirs(path)
    except: 
        shutil.rmtree(path)
        os.makedirs(path)

    return SummaryWriter(path)
    