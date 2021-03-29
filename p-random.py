from models.vgg import vgg16_bn

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

if __name__ == '__main__':
    ## vgg16모델 만들고 저번에 저장한 파일에서 state_dict 불러와서 저장
    vgg = vgg16_bn()
    vgg.load_state_dict(torch.load("checkpoint/vgg16/Saturday_13_March_2021_20h_29m_30s/vgg16-195-best.pth"))
    vgg.eval()
    parameters_to_prune=[]

    ## relu나 dropout 같은게 들어가면 안되니까

    for i in vgg.features:
        if isinstance(i,torch.nn.modules.conv.Conv2d):
            parameters_to_prune.append([i,'weight'])
    
    for i in vgg.classifier:
        if isinstance(i,torch.nn.modules.linear.Linear):
            parameters_to_prune.append([i,'weight'])

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=0.2
    )
    
    psum=0
    sum=0

    ## sparsity 구하기!
    for i in vgg.features:
        if isinstance(i,torch.nn.modules.conv.Conv2d):
            psum+=torch.sum(i.weight==0)
            sum+=i.weight.nelement()
            prune.remove(i, 'weight')

    for i in vgg.classifier:
        if isinstance(i,torch.nn.modules.linear.Linear):
            psum+=torch.sum(i.weight==0)
            sum+=i.weight.nelement()
            prune.remove(i, 'weight')
    
    print(
        "Global sparsity: {:.2f}%".format(
        100. * float(psum)
        / float(sum)
        )
    )




    print(list(vgg.named_parameters()))
    





