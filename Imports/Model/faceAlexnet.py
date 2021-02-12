import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import torch.utils.data
import math
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out
'''
class AlexNet_Feature_Extractor(nn.Module):
    #for feature loss
    def __init__(self,pretrainded=False,modelpath=None):
        super(AlexNet_Feature_Extractor, self).__init__()
        assert pretrainded is False or modelpath is not None,"pretrain model need to be specified"
        self.relu=nn.ReLU()
        self.Maxpool=nn.MaxPool2d(3,2)
        self.Conv1=nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.Conv2=nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.Conv3=nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.Conv4=nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.Conv5=nn.Conv2d(256, 256, kernel_size=3, padding=1)

        if pretrainded is True:
            self.load_pretrained_params(modelpath)

    def forward(self, x):
        conv1=self.Conv1(x)
        pool1=self.Maxpool(self.relu(conv1))

        conv2 = self.Conv2(pool1)
        pool2 = self.Maxpool(self.relu(conv2))

        conv3 = self.Conv3(pool2)
        relu3 = self.relu(conv3)

        conv4=self.Conv4(relu3)
        relu4=self.relu(conv4)

        conv5=self.Conv5(relu4)
        return conv5

    def load_pretrained_params(self,path):
        # step1: load pretrained model
        pretrained_dict = torch.load(path)
        # step2: get model state_dict
        model_dict = self.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.load_state_dict(model_dict)
'''




class Img_to_zero_center(object):
    def __int__(self):
        pass
    def __call__(self, t_img):
        '''
        :param img:tensor be 0-1
        :return:
        '''
        t_img=(t_img-0.5)*2
        return t_img



class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)



class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return _conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# custom conv2d, because pytorch don't have "padding='same'" option.
def _conv2d_same_padding(input, weight, bias=None, stride=(1,1), padding=1, dilation=(1,1), groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class AgeAlexNet(nn.Module):
    def __init__(self,pretrainded=False,modelpath=None):
        super(AgeAlexNet, self).__init__()
        assert pretrainded is False or modelpath is not None,"pretrain model need to be specified"
        self.features = nn.Sequential(
            Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2,2e-5,0.75),

            Conv2d(96, 256, kernel_size=5, stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2, 2e-5, 0.75),

            Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            Conv2d(384, 384, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),

            Conv2d(384, 256, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.age_classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),
        )
        if pretrainded is True:
            self.load_pretrained_params(modelpath)

        self.Conv3_feature_module=nn.Sequential()
        self.Conv4_feature_module=nn.Sequential()
        self.Conv5_feature_module=nn.Sequential()
        self.Pool5_feature_module=nn.Sequential()
        for x in range(10):
            self.Conv3_feature_module.add_module(str(x), self.features[x])
        for x in range(10,12):
            self.Conv4_feature_module.add_module(str(x),self.features[x])
        for x in range(12,14):
            self.Conv5_feature_module.add_module(str(x),self.features[x])
        for x in range(14,15):
            self.Pool5_feature_module.add_module(str(x),self.features[x])


    def forward(self, x):
        self.conv3_feature=self.Conv3_feature_module(x)
        self.conv4_feature=self.Conv4_feature_module(self.conv3_feature)
        self.conv5_feature=self.Conv5_feature_module(self.conv4_feature)
        pool5_feature=self.Pool5_feature_module(self.conv5_feature)
        self.pool5_feature=pool5_feature
        flattened = pool5_feature.view(pool5_feature.size(0), -1)
        age_logit = self.age_classifier(flattened)
        return age_logit

    def load_pretrained_params(self,path):
        # step1: load pretrained model
        pretrained_dict = torch.load(path)
        # step2: get model state_dict
        model_dict = self.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.load_state_dict(model_dict)


class AgeClassify:
    def __init__(self):
        #step 1:define model
        self.model=AgeAlexNet(pretrainded=False).cuda()
        #step 2:define optimizer
        self.optim=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(0.5, 0.999))
        #step 3:define loss
        self.criterion=nn.CrossEntropyLoss().cuda()

    def train(self,input,label):
        self.model.train()
        output=self.model(input)
        self.loss=self.criterion(output,label)

    def val(self,input):
        self.model.eval()
        output=F.softmax(self.model(input),dim=1).max(1)[1]
        return output

    def save_model(self,dir,filename):
        torch.save(self.model.state_dict(),os.path.join(dir,filename))
        
        
        

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = Conv2d(69, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = Conv2d(512, 512, kernel_size=4, stride=2)

    def forward(self, x,condition):
        x = self.lrelu(self.conv1(x))
        x=torch.cat((x,condition),1)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = Conv2d(32, 3, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.bn4 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn5 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.repeat_blocks=self._make_repeat_blocks(BasicBlock(128,128),6)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0,output_padding=1)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def _make_repeat_blocks(self,block,repeat_times):
        layers=[]
        for i in range(repeat_times):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x,condition=None):
        if condition is not None:
            x=torch.cat((x,condition),1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.repeat_blocks(x)
        x = self.deconv1(x)
        x = self.relu(self.bn4(x))
        x = self.deconv2(x)
        x = self.relu(self.bn5(x))
        x = self.tanh(self.conv4(x))
        return x

class IPCGANs:
    def __init__(self,lr=0.01,age_classifier_path=None,gan_loss_weight=75,feature_loss_weight=0.5e-4,age_loss_weight=30):

        self.d_lr=lr
        self.g_lr=lr

        self.generator=Generator().cuda()
        self.discriminator=PatchDiscriminator().cuda()
        if age_classifier_path is not None:
            self.age_classifier=AgeAlexNet(pretrainded=True,modelpath=age_classifier_path).cuda()
        else:
            self.age_classifier = AgeAlexNet(pretrainded=False).cuda()
        self.MSEloss=nn.MSELoss().cuda()
        self.CrossEntropyLoss=nn.CrossEntropyLoss().cuda()

        self.gan_loss_weight=gan_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.age_loss_weight=age_loss_weight

        self.d_optim = torch.optim.Adam(self.discriminator.parameters(),self.d_lr,betas=(0.5,0.99))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.99))

    def save_model(self,dir,filename):
        torch.save(self.generator.state_dict(),os.path.join(dir,"g"+filename))
        torch.save(self.discriminator.state_dict(),os.path.join(dir,"d"+filename))

    def load_generator_state_dict(self,state_dict):
        pretrained_dict = state_dict
        # step2: get model state_dict
        model_dict = self.generator.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.generator.load_state_dict(model_dict)

    def test_generate(self,source_img_128,condition):
        self.generator.eval()
        with torch.no_grad():
            generate_image=self.generator(source_img_128,condition)
        return generate_image

    def cuda(self):
        self.generator=self.generator.cuda()

    def train(self,source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, age_label):
        '''

        :param source_img_227: use this img to extract conv5 feature
        :param source_img_128: use this img to generate face in target age
        :param true_label_img:
        :param true_label_128:
        :param true_label_64:
        :param fake_label_64:
        :param age_label:
        :return:
        '''

        ###################################gan_loss###############################
        self.g_source=self.generator(source_img_128,condition=true_label_128)

        #real img, right age label
        #logit means prob which hasn't been normalized

        #d1 logit ,discriminator 1 means true,0 means false.
        d1_logit=self.discriminator(true_label_img,condition=true_label_64)

        d1_real_loss=self.MSEloss(d1_logit,torch.ones((d1_logit.size())).cuda())

        #real img, false label
        d2_logit=self.discriminator(true_label_img,condition=fake_label_64)
        d2_fake_loss=self.MSEloss(d2_logit,torch.zeros((d1_logit.size())).cuda())

        #fake img,real label
        d3_logit=self.discriminator(self.g_source,condition=true_label_64)
        d3_fake_loss=self.MSEloss(d3_logit,torch.zeros((d1_logit.size())).cuda())#use this for discriminator
        d3_real_loss=self.MSEloss(d3_logit,torch.ones((d1_logit.size())).cuda())#use this for genrator

        self.d_loss=(1./2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))) * self.gan_loss_weight
        g_loss=(1./2*d3_real_loss)*self.gan_loss_weight


        ################################feature_loss#############################

        self.age_classifier(source_img_227)
        source_feature=self.age_classifier.conv5_feature

        generate_img_227 = F.interpolate(self.g_source, (227, 227), mode="bilinear", align_corners=True)
        generate_img_227 = Img_to_zero_center()(generate_img_227)

        self.age_classifier(generate_img_227)
        generate_feature =self.age_classifier.conv5_feature
        self.feature_loss=self.MSEloss(source_feature,generate_feature)

        ################################age_cls_loss##############################



        age_logit=self.age_classifier(generate_img_227)
        self.age_loss=self.CrossEntropyLoss(age_logit,age_label)

        self.g_loss=self.age_loss+g_loss+self.feature_loss

        
if __name__=="__main__":
    
    tensor=torch.ones((2,3,128,128)).cuda()
    condition=torch.ones((2,5,64,64)).cuda()

    discriminator=PatchDiscriminator().cuda()
    from tensorboardX import SummaryWriter
    print(discriminator(tensor,condition).size())
    with SummaryWriter(comment='Net1')as w:
        w.add_graph(discriminator, (tensor,condition))
    dsa
    
    tensor=torch.ones((1,8,128,128))
    conv2d=Conv2d(8, 32, kernel_size=3, stride=2)
    print(conv2d(tensor).size())
    one=torch.ones((1,3,227,227))
    model=AgeAlexNet()