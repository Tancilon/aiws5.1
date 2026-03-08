import copy
import hydra
import pathlib
import shutil
from omegaconf import OmegaConf
from termcolor import cprint
from hydra.core.hydra_config import HydraConfig

from components.category_recognition import CategoryRecognition
from components.dimension_measurement import DimensionMeasurement
from components.pose_estimation import PoseEstimation

class Workspace:
    def __init__(self,
                 cfg: OmegaConf):
        self.cfg = cfg

    
    def clean_data(self, cfg: OmegaConf):
        tmp_dir = pathlib.Path(cfg.tmp_dir)
        if tmp_dir.exists():
            for item in tmp_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        if not cfg.clean_hydra_output:
            return

        hydra_cfg = HydraConfig.get()
        run_dir = pathlib.Path(hydra_cfg.run.dir)
        parent_dir = run_dir.parent
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        cprint("[Workspace] Clenning all files in tmp_dir...", "yellow")
        self.clean_data(cfg)

        cprint("[Workspace] Running category recognition algorithm...", "yellow")
        self.category_recognition: CategoryRecognition = hydra.utils.instantiate(cfg.category_recognition)
        cat_result = self.category_recognition.infer(cfg.rgb_path)
        if cfg.debug: print(f"[Debug] cat_result: {cat_result}")

        cprint("[Workspace] Running dimension measurement algorithm...", "yellow")
        self.dimension_measurement: DimensionMeasurement = hydra.utils.instantiate(cfg.dimension_measurement) 
        dim_result = self.dimension_measurement.infer(cfg.rgb_path, cfg.depth_path, cat_result[0]['class_name'])
        if cfg.debug: print(f"[Debug] dim_result: {dim_result}")

        cprint("[Workspace] Running pose estimation algorithm...", "yellow") 
        self.pose_estimation: PoseEstimation = hydra.utils.instantiate(cfg.pose_estimation)
        pose_resuilt = self.pose_estimation.infer(cfg.rgb_path, cfg.depth_path, cat_result[0]['class_name'], dim_result[0] if cfg.dimension_measurement.query_mode else dim_result)
        if cfg.debug: print(f"[Debug] pose_result: {pose_resuilt}")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config'))
)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__=='__main__':
    main()
