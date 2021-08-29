from option import Option
from custom_dataloader import load_data
from models.OCGAN.OCGAN import OCgan
from tqdm import tqdm

if __name__=='__main__':
    opt = Option().parse()
    normal_classes = [8]
    dataloader = load_data(opt,normal_classes,train=True)

    model = OCgan(opt)
    
    best_acc = 0.0 # best accuracy

    for epoch in tqdm(range(opt.n_epochs),ncols=100):
        for inputs, labels in dataloader:
            model.set_input(inputs, labels)
            model.train()