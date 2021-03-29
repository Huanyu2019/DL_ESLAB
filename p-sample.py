from models.vgg import vgg16_bn

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F


if __name__ == '__main__':

    ## vgg16모델 만들고 저번에 저장한 파일에서 state_dict 불러와서 저장
    vgg = vgg16_bn()
    vgg.load_state_dict(torch.load('checkpoint/vgg16/Saturday_13_March_2021_20h_29m_30s/vgg16-195-best.pth'))
    vgg.eval()


    ## feature[0] : 즉 첫번째 conv layer만 pruning하기!
    module=vgg.features[0]
    print(list(module.named_parameters()))
    prune.random_unstructured(module, name="weight", amount=0.3)
    ## weight라는 이름의 파라미터의 30%를 랜덤으로 가지치기하기
    print(list(module.named_buffers()))
    ## weight_mask라는 이름으로 mask가 저장됨
    ##가지치기 기법은 파이토치의 forward_pre_hooks 를 이용하여 각 순전파가 진행되기 전에 가지치기 기법이 적용됨 --??

    print(module.weight)
    ## 와! 가중치가 pruning 됐다!

    prune.remove(module, 'weight')
    ## pruning된 weight를 진짜 파라미터로 설정하기

    print(list(module.named_parameters()))
    ## 와! feature[0]이 pruning되었다!