from monai.networks.nets.resnet import ResNet,ResNetBlock,ResNetBottleneck
from monai.networks.nets.densenet import DenseNet121
from monai.networks.nets.efficientnet import EfficientNetBN
from torchvision.models.mobilenetv2 import MobileNetV2_pancreas
from torchvision.models.mobilenetv3 import mobilenet_v3_pancreas
from torchvision.models.convnext import convnext_pancreas
from torchvision.models.swin_transformer import swin_t_pancreas,swin_v2_t_pancreas
from torchvision.models.vision_transformer import vit_b_16_pancreas
def get_model(name:str,num_class:int):
    if name == 'resnet50':
        model = ResNet(block=ResNetBottleneck,layers=[3, 4, 6, 3],block_inplanes=[64, 128, 256, 512],spatial_dims=2, n_input_channels=1, num_classes=num_class)
    elif name=='resnet34':
        model = ResNet(block=ResNetBlock,layers=[3, 4, 6, 3],block_inplanes=[64, 128, 256, 512],spatial_dims=2, n_input_channels=1, num_classes=num_class)
    elif name=='densnet121':
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class)
    elif name=='efficientnet':
        model = EfficientNetBN(model_name='efficientnet-b0',pretrained=False,spatial_dims=2, in_channels=1, num_classes=num_class)
    elif name=='vit':
        model = vit_b_16_pancreas(num_classes=2,weights=None,progress=False)
    elif name=='mobilenetv2':
        model = MobileNetV2_pancreas(num_classes=num_class)
    elif name =='mobilenetv3_s':
        model = mobilenet_v3_pancreas(arch='mobilenet_v3_small',num_classes=num_class,weights=None,progress=False)
    elif name =='mobilenetv3_l':
        model = mobilenet_v3_pancreas(arch='mobilenet_v3_large',num_classes=num_class,weights=None,progress=False)
    elif name =='convnext_tiny':
        model = convnext_pancreas(arch='convnext_tiny',num_classes=num_class,weights=None,progress=False)
    elif name =='convnext_small':
        model = convnext_pancreas(arch='convnext_small',num_classes=num_class,weights=None,progress=False)
    elif name =='swin_t':
        model = swin_t_pancreas(num_classes=num_class,weights=None,progress=False)
    elif name =='swin_v2_t':
        model = swin_v2_t_pancreas(num_classes=num_class,weights=None,progress=False)
    else:
        print("please enter correct model name")
    return model
