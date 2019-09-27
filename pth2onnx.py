import argparse
import torch.onnx
from torch.autograd import Variable

from baseline.model.DeepMAR import DeepMAR_ResNet50


parser = argparse.ArgumentParser(description="attribute detection")
parser.add_argument("--trained_model", type=str)
# parser.add_argument("--label_file", type=str, help="The label file path.")
# parser.add_argument("--nms_method", type=str, default="hard")
args = parser.parse_args()

model_kwargs = {'drop_pool5': True,
                 'drop_pool5_rate': 0.5,
                 'last_conv_stride': 2,
                 'num_att': 35}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    model = DeepMAR_ResNet50(**model_kwargs)
    state = torch.load('/home/thulab/Desktop/pedestrian-attribute-recognition-pytorch/exp/deepmar_resnet50/pa100k/partition0/run1/model/ckpt_epoch150.pth')
    # print(type(state['state_dicts']))
    # print(len(state['state_dicts']))
    # print(state['state_dicts'][0])
    model.load_state_dict(state['state_dicts'][0], strict=True)

    example = torch.rand(1, 3, 224, 224)

    torch_out = torch.onnx.export(model,
                                  example,
                                  "test.onnx",
                                  verbose=True
                                  )

    print('onnx done!')

