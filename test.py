from torch.autograd import Variable
import argparse
from math import log10
from model import model
from model.model import weights_init
from utils import *
from PIL import Image
from data import get_test_dataloader

parser = argparse.ArgumentParser(description='Super Resolution')

# validation data
parser.add_argument('--HR_valDataroot', required=False, default='data/benchmark/Set5/HR') # modifying to your SR_data folder path
parser.add_argument('--LR_valDataroot', required=False, default='data/benchmark/Set5/LR_bicubic/X2') # modifying to your SR_data folder path
parser.add_argument('--valBatchSize', type=int, default=5)

parser.add_argument('--pretrained_model', default='result/Net1/model/model_lastest.pt', help='save result')


parser.add_argument('--nRG', type=int, default=3, help='number of RG block')
parser.add_argument('--nRCAB', type=int, default=2, help='number of RCAB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=64, help='patch size')

parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')

parser.add_argument('--scale', type=float, default=2, help='scale output size /input size')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def test(args):

    # SR network
    my_model = model.EDSR()
    my_model.apply(weights_init)
    my_model.cuda()

    my_model.load_state_dict(torch.load(args.pretrained_model))

    testdataloader = get_test_dataloader('Set5',args)
    my_model.eval()

    avg_psnr = 0
    avg_ssim = 0
    count = 0
    for batch, (im_lr, im_hr) in enumerate(testdataloader):
        count = count + 1
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = my_model(im_lr)

        output = unnormalize(output)
        out = Image.fromarray(np.uint8(output),mode='RGB')  # output of SRCNN
        out.save('result/Net1/SR_img_%03d.png' % (count))

        # =========== Target Image ===============
        im_hr = unnormalize(im_hr)

        psnr , ssim = psnr_ssim_from_sci(output,im_hr)
        print( '%d_img PSNR/SSIM: %.4f/%.4f '%(count,psnr,ssim))
        avg_ssim += ssim
        avg_psnr += psnr

    print('AVG PSNR/AVG SSIM : %.4f/%.4f '%(avg_psnr/len(testdataloader.dataset),avg_ssim / len(testdataloader.dataset)))


if __name__ == '__main__':
    test(args)
