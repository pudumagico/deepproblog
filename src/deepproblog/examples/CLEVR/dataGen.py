import torch
from tqdm import tqdm

from yolo.detect import create_data_loader
# from yolo.models import Darknet
from yolo.models import load_model
from yolo.utils.utils import load_classes, non_max_suppression


def termPath2dataList(termPath, img_size, domain):
    """
    @param termPath: a string of the form 'term path' denoting the path to the files represented by term
    """
    factsList = []
    dataList = []
    # Load Yolo network, which is used to generate the facts for bounding boxes and a tensor for each bounding box
    config_path = './yolo/yolov3_scene_parser.cfg'
    weights_path = './yolo/yolov3_ckpt_200.pth'
    yolo = load_model(config_path, weights_path)
    yolo.eval()

    # feed each image into yolo
    term, path = termPath.split(' ')
    dataloader = create_data_loader(path, 1, img_size, 1)
    
    for _, img in tqdm(dataloader):
        # img = Variable(img.type(torch.FloatTensor))
        img = img.to("cuda")
        with torch.no_grad():
            output = yolo(img)

            facts, dataDic = postProcessing(output, term, domain)
            factsList.append(facts)
            dataList.append(dataDic)
    return factsList, dataList


def postProcessing(output, term, domain, num_classes=96, conf_thres=0.3, nms_thres=0.4):
    facts = ''
    dataDic = {}
    cls_name = load_classes('./yolo/clver.names')
    detections = non_max_suppression(output, conf_thres, nms_thres)

    if detections:
        for detection in detections:
            for idx, (x1, y1, x2, y2, cls_conf, cls_pred) in enumerate(detection):
                terms = '{},b{}'.format(term, idx)
                facts += 'box({}, {}, {}, {}, {}).\n'.format(terms, max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2)))
                className = '{}'.format(cls_name[int(cls_pred)])
                X = torch.zeros([1, len(domain)], dtype=torch.float64)
                if className in domain:
                    X[0, domain.index(className)] += round(float(cls_conf), 3)
                else:
                    X[0, -1] += round(float(cls_conf), 3)
                dataDic[terms] = X
    return facts, dataDic
