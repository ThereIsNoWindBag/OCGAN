import os
import shutil
import numpy as np
import torch
import torch.nn as nn

from models.OCGAN.networks import Encoder, Decoder
from models.OCGAN.networks import Discriminator_l
from models.OCGAN.networks import Discriminator_v
from models.OCGAN.networks import Classifier

from utils.utils import weights_init

from models.OCGAN.evaluation import evaluate

class OCgan():
    def __init__(self,opt):
        self.opt = opt

        #networks init
        self.net_enc = Encoder(opt)
        self.net_dec = Decoder(opt)
        self.net_D_l = Discriminator_l(opt)
        self.net_D_v = Discriminator_v(opt)
        self.net_clf = Classifier(opt)

        self.net_enc.apply(weights_init)
        self.net_dec.apply(weights_init)
        self.net_D_l.apply(weights_init)
        self.net_D_v.apply(weights_init)
        self.net_clf.apply(weights_init)

        self.net_enc.cuda()
        self.net_dec.cuda()
        self.net_D_l.cuda()
        self.net_D_v.cuda()
        self.net_clf.cuda()

        # #variable init
        # self.input = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
        #                             dtype=torch.float32, device= torch.device(self.opt.device))
        # self.l2 = torch.empty(size= (self.opt.batchsize,self.opt.latent_dim,self.opt.latent_size),
        #                             dtype=torch.float32, device= torch.device(self.opt.device),requires_grad=True)
        # self.l1 = torch.empty(size = (self.opt.batchsize,self.opt.latent_dim, self.opt.latent_size))
        # self.label = torch.empty(size= (self.opt.batchsize,),
        #                             dtype=torch.float32, device= torch.device(self.opt.device))
        # self.rec_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
        #                             dtype=torch.float32, device= torch.device(self.opt.device))
        # self.fake_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
        #                             dtype=torch.float32, device= torch.device(self.opt.device))

        # self.fixed_input = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
        #                             dtype=torch.float32, device= torch.device(self.opt.device))
        # self.fixed_rec_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
        #                             dtype=torch.float32, device= torch.device(self.opt.device))
        
        #optimizer
        self.optimizer_enc = torch.optim.Adam(self.net_enc.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_dec = torch.optim.Adam(self.net_dec.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_D_l = torch.optim.Adam(self.net_D_l.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_D_v = torch.optim.Adam(self.net_D_v.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_clf = torch.optim.Adam(self.net_clf.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        # self.optimizer_l2 = torch.optim.Adam([{'params': self.l2}], lr =self.opt.lr, betas=(0.9,0.99))

        #criterion
        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_l1_norm = nn.L1Loss().cuda()
        self.criterion_bce = nn.BCELoss().cuda()

    def set_input(self, input, label):
        self.input = input.cuda()
        self.label = label.cuda()

    def train(self):
        ## Classifier Update
        n = torch.randn(self.opt.batchsize, self.opt.n_channels, self.opt.isize, self.opt.isize, requires_grad=False).cuda()
        l1 = self.net_enc(self.input + (n * 0.2))

        u = np.random.uniform(-1, 1, (self.opt.batchsize, self.opt.latent_size))   
        l2 = torch.from_numpy(u).float().cuda()

        dec_l1 = self.net_dec(l1)
        dec_l2 = self.net_dec(l2)

        logits_C_real = self.net_clf(dec_l1)
        logits_C_fake = self.net_clf(dec_l2)

        real_logits_C = torch.ones([logits_C_real.shape[0], 1]).cuda()
        fake_logits_C = torch.zeros([logits_C_real.shape[0], 1]).cuda()

        loss_cl_real = self.criterion_bce(logits_C_real, real_logits_C)
        loss_cl_fake = self.criterion_bce(logits_C_fake, fake_logits_C)

        loss_cl = loss_cl_real + loss_cl_fake

        self.net_clf.zero_grad()
        loss_cl.backward(retain_graph=True)
        self.optimizer_clf.step()

        # Discriminator update
        logits_D1_l1 = self.net_D_l(l1)
        logits_D1_l2 = self.net_D_l(l2)

        label_real_Dl_l1 = torch.ones([logits_C_real.shape[0], 1]).cuda()
        label_fake_Dl_l2 = torch.zeros([logits_C_real.shape[0], 1]).cuda()
        
        loss_Dl_l1  = self.criterion_bce(logits_D1_l1, label_real_Dl_l1)
        loss_Dl_l2  = self.criterion_bce(logits_D1_l2, label_fake_Dl_l2)

        loss_DL = loss_Dl_l1 + loss_Dl_l2

        logits_Dv_real = self.net_D_v(self.input)
        fake_img = self.net_dec(l2)
        logits_Dv_fake =self.net_D_v(fake_img)

        label_real_Dv = torch.ones([logits_C_real.shape[0], 1]).cuda()
        label_fake_Dv = torch.zeros([logits_C_real.shape[0], 1]).cuda()
        
        loss_Dv_real = self.criterion_bce(logits_Dv_real, label_real_Dv)
        loss_Dv_fake = self.criterion_bce(logits_Dv_fake, label_fake_Dv)
        
        loss_Dv = loss_Dv_real + loss_Dv_fake

        loss_total = loss_DL + loss_Dv

        self.net_D_v.zero_grad()
        self.net_D_l.zero_grad()
        loss_total.backward()
        self.optimizer_D_v.step()
        self.optimizer_D_l.step()

        # Informative-negative mining
        for i in range(5):
            logits_c_l2_mine = self.net_clf(self.net_dec(self.l2))
            fake_label_mine = torch.ones([logits_C_real.shape[0], 1]).cuda()
            loss_mine = self.criterion_bce(logits_c_l2_mine,fake_label_mine)
            self.optimizer_l2.zero_grad()
            loss_mine.backward()
            self.optimizer_l2.step()

        # Generator(Encoder + Decoder) update
        self.rec_img = self.net_dec(l1)
        self.fake_img =self.net_dec(l2)
        
        self.loss_mse = self.criterion_mse(self.rec_img, self.input)
        
        label_real_dl_ae = torch.ones([logits_C_fake.shape[0], 1]).cuda()
        
        self.loss_AE_l = self.criterion_bce(logits_D1_l1,label_real_dl_ae)
        logits_Dv_l2_mine = self.net_D_v(dec_l2)
        zeros_logits_Dv_l2_mine = torch.zeros([logits_Dv_l2_mine.shape[0], 1]).cuda()

        self.loss_AE_v = self.criterion_bce(logits_Dv_l2_mine, zeros_logits_Dv_l2_mine)

        self.loss_ae_all = 10 * self.loss_mse + self.loss_AE_v + self.loss_AE_l

        self.net_enc.zero_grad()
        self.net_dec.zero_grad()
        self.loss_ae_all.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()

        # print(f'\rloss_AE_v: {loss_AE_v} loss_AE_l: {loss_AE_l} loss_AE_all: {loss_ae_all}',end='')
    
    def evaluate(self, dataloader, epoch):
        self.net_dec.eval()
        self.net_enc.eval()

        with torch.no_grad():
            an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=torch.device('cpu'))
            gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long,device=torch.device('cpu'))
        # an_scores = np.zeros(shape=(len(dataloader.dataset,)))
        # gt_labels = np.zeros(shape=(len(dataloader.dataset,)))

        for i, (inputs,labels) in enumerate(dataloader):
            self.set_input(inputs,labels)
            # self.input = inputs.cuda()
            latent_i = self.net_enc(self.input)
            self.fake_img = self.net_dec(latent_i)
            input= self.input.view([self.opt.batchsize,-1])
            cpu_fake_img =  self.fake_img.view([self.opt.batchsize,-1])
            self.error = torch.mean(torch.pow((input - cpu_fake_img),2),dim=1).detach().cpu()

            # self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize + len(self.error)] = self.error.reshape(self.error.size(0))
            an_scores[i*self.opt.batchsize : i*self.opt.batchsize + len(self.error)] = self.error
            gt_labels[i*self.opt.batchsize : i*self.opt.batchsize + self.error.size(0)] = labels #.view(self.error.size)

        # an_scores = (an_scores - torch.min(an_scores))/(torch.max(an_scores)-torch.min(an_scores))
        an_scores = (an_scores - torch.min(an_scores))/(torch.max(an_scores)-torch.min(an_scores))
        self.auc, self.thres_hold = evaluate(gt_labels, an_scores)

        self.vis.plot_current_acc(epoch,self.opt.test_ratio,self.auc)

        self.net_dec.train()
        self.net_enc.train()

        return self.auc

    def save_weight(self, epoch):
        if not os.path.exists(self.opt.weight_path):
            os.mkdir(self.opt.weight_path)

        # Save each component
        torch.save({
                    'net_dec': self.net_dec.state_dict(),
                    'net_enc': self.net_enc.state_dict(),
                    'net_D_l': self.net_D_l.state_dict(),
                    'net_D_v': self.net_D_v.state_dict(),
                    'net_clf': self.net_clf.state_dict(),
                    'epoch': epoch
                    }
                    , os.path.join(self.opt.weight_path, f'{epoch}_weight.pt'))
        shutil.copy(os.path.join(self.opt.weight_path, f'{epoch}_weight.pt'), os.path.join(self.opt.weight_path, f'latest.pt'))

    def load_weight(self):
        checkpoint = torch.load(os.path.join(self.opt.weight_path, f'latest.pt'))

        self.net_dec.load_state_dict(checkpoint['net_dec'])
        self.net_enc.load_state_dict(checkpoint['net_enc'])
        self.net_D_l.load_state_dict(checkpoint['net_D_l'])
        self.net_D_v.load_state_dict(checkpoint['net_D_v'])
        self.net_clf.load_state_dict(checkpoint['net_clf'])
        

    def visual(self,l2=False):
        if not l2:
            self.vis.display_current_images(self.input,self.rec_img)
        else:
            self.vis.display_current_images(self.input,self.rec_img,self.fake_img)
    
    def visual_test(self):
        self.vis.display_fixed_images(self.fixed_input,self.fixed_rec_img)
        
        








