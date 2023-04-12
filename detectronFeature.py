import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# 基本库
import json

# detectrion库的模块加载
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg




# 将json数据加载到字典中
def get_board_dicts(imgdir):
    json_file = imgdir+"/dataset.json"
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"] 
        i["file_name"] = imgdir+"/"+filename 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYWH_ABS
            j["category_id"] = int(j["category_id"])
    return dataset_dicts


# 继承detectron的trainer来加载配置
class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def train(dataName,tags,modelPath,jsonPath):
    # 注册训练集和验证集
    # register的第一个参数是注册的数据集的名字,设置为数据集名字+_+train/val
    for d in ["train", "val"]:
        DatasetCatalog.register(dataName+'_'+ d, lambda d=d: get_board_dicts(jsonPath + d))
        MetadataCatalog.get(dataName + d).set(thing_classes=tags)
    board_metadata = MetadataCatalog.get(dataName)
    # 模型的modelPath指的是模型对应的Key
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(modelPath))
    cfg.DATASETS.TRAIN = (dataName+'_'+'train',)
    cfg.DATASETS.TEST = (dataName+'_'+'val',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(modelPath) 
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0125  
    cfg.SOLVER.MAX_ITER = 1500   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
     
if __name__=='__main__':
    train('diffraction',
          ["diffraction"],
          'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
          './annotation/'
          )