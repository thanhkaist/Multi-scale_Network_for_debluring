from torch.autograd import Variable
import argparse
from math import log10
from model import model
from model.model import weights_init
from utils import *
from PIL import Image
from data import get_test_dataloader

parser = argparse.ArgumentParser(description='Super Resolution')

parser.add_argument('model_type', choices=['one_scale', 'one_scale_lsc', 'multi_scale', 'multi_scale_lsc'])
parser.add_argument('--saveDir', default='GoPro', help='datasave directory')

# validation data
parser.add_argument('--HR_valDataroot', required=False,
                    default='data/benchmark/Set5/HR')  # modifying to your SR_data folder path
parser.add_argument('--LR_valDataroot', required=False,
                    default='data/benchmark/Set5/LR_bicubic/X2')  # modifying to your SR_data folder path
parser.add_argument('--valBatchSize', type=int, default=5)

parser.add_argument('--pretrained_model', default='multi_scale_lsc1000/Net1/model/model_best.pt', help='save result')

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
    one_scale = False

    if args.model_type == 'one_scale':
        my_model = model.OneScale(3, False)
        one_scale = True
    elif args.model_type == 'one_scale_lsc':
        my_model = model.OneScale(3, True)
        one_scale = True
    elif args.model_type == 'multi_scale':
        my_model = model.MultiScale(False)
    elif args.model_type == 'multi_scale_lsc':
        my_model = model.MultiScale(True)
    else:
        raise Exception("Model type is not supported: {}".format(args.model_type))

    my_model.apply(weights_init)
    my_model.cuda()

    my_model.load_state_dict(torch.load(args.pretrained_model))

    testdataloader = get_test_dataloader('GoPro', args)
    my_model.eval()

    avg_psnr = 0
    avg_ssim = 0
    avg_msssim = 0
    count = 0

    # make val folder
    if not os.path.isdir("val/%s/%s"%(args.model_type,args.saveDir)):
        os.makedirs("val/%s/%s"%(args.model_type,args.saveDir),exist_ok=False)

    logFile = open("val/%s/%s"%(args.model_type,args.saveDir) + '/log.txt', 'w')

    for batch, images in enumerate(testdataloader):
        with torch.no_grad():
            blur_img_s1 = images['blur_image_s1']
            blur_img_s2 = images['blur_image_s2']
            blur_img_s3 = images['blur_image_s3']
            sharp_img_s1 = images['sharp_image_s1']
            sharp_img_s2 = images['sharp_image_s2']
            sharp_img_s3 = images['sharp_image_s3']

            blur_img_s1 = Variable(blur_img_s1.cuda())
            blur_img_s2 = Variable(blur_img_s2.cuda())
            blur_img_s3 = Variable(blur_img_s3.cuda())
            sharp_img_s1 = Variable(sharp_img_s1.cuda())
            sharp_img_s2 = Variable(sharp_img_s2.cuda())
            sharp_img_s3 = Variable(sharp_img_s3.cuda())
            if one_scale:
                output = my_model(blur_img_s1)
            else:
                output, _, _ = my_model(blur_img_s1, blur_img_s2, blur_img_s3)

        output = unnormalize(output[0])
        im_hr = unnormalize(sharp_img_s1[0])
        psnr, ssim = psnr_ssim_from_sci(output, im_hr)
        avg_psnr += psnr
        avg_ssim += ssim
        count = count + 1
        out = Image.fromarray(np.uint8(output), mode='RGB')
        out.save("val/%s/%s/DB_img_%03d.png"%(args.model_type,args.saveDir,count))

        # =========== Target Image ===============
        psnr, ssim = psnr_ssim_from_sci(output, im_hr, padding=0, y_channels=False)
        msssim = MultiScaleSSIM(output[None], im_hr[None])

        log = '%d_img PSNR/SSIM/MS-SSIM: %.4f/%.4f/%.4f ' % (count, psnr, ssim, msssim)
        print(log)
        logFile.write(log + '\n')
        logFile.flush()
        avg_ssim += ssim
        avg_psnr += psnr
        avg_msssim += msssim

    log = 'AVG PSNR/AVG SSIM/AVG MS-SSIM : %.4f/%.4f/%.4f ' % (
    avg_psnr / len(testdataloader.dataset), avg_ssim / len(testdataloader.dataset),
    avg_msssim / len(testdataloader.dataset))
    print(log)
    logFile.write(log + '\n')
    logFile.flush()

if __name__ == '__main__':
    test(args)
