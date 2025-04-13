from pydantic import BaseModel
from typing import Union
import yaml


# ===== BaseConfig: chỉ để dùng lại =====
class BaseConfigs(BaseModel):
    dataroot: str
    batch_size: int
    loadSize: int
    fineSize: int
    input_nc: int
    output_nc: int
    ngf: int
    ndf: int
    netD: str
    netG: str
    n_layers_D: int
    gpu_ids: str
    name: str
    dataset_mode: str
    model: str
    epoch: str
    num_threads: int
    checkpoints_dir: str
    norm: str
    serial_batches: bool
    no_dropout: bool
    max_dataset_size: Union[int, float]
    resize_or_crop: str
    no_flip: bool
    init_type: str
    init_gain: float
    verbose: bool
    suffix: str
    no_patch: bool


# ===== Train config =====
class TrainConfigs(BaseConfigs):
    display_freq: int
    display_ncols: int
    display_id: int
    display_server: str
    display_env: str
    display_port: int
    update_html_freq: int
    print_freq: int
    save_latest_freq: int
    save_epoch_freq: int
    continue_train: bool
    epoch_count: int
    phase: str
    niter: int
    niter_decay: int
    beta1: float
    lr: float
    no_lsgan: bool
    lambda_A: float
    no_html: bool
    lr_policy: str
    lr_decay_iters: int
    w_pa: float
    w_la: float
    w_co: float
    train_imagenum: int


# ===== Test config =====
class TestConfigs(BaseConfigs):
    ntest: Union[int, float]
    results_dir: str
    aspect_ratio: float
    phase: str
    num_test: int
    interval: float
    eval: bool
    w_pa: float
    w_la: float
    w_co: float


# ===== Hàm load YAML =====
def load_config(yaml_path: str, mode: str = "") -> Union[TrainConfigs, TestConfigs, BaseConfigs]:
    with open(yaml_path, "r") as f:
        full_cfg = yaml.safe_load(f)

    base_cfg = full_cfg.get("baseconfigs", {})
    if mode == "train":
        train_cfg = full_cfg.get("trainconfigs", {})
        merged_cfg = {**base_cfg, **train_cfg}
        return TrainConfigs(**merged_cfg)
    elif mode == "test":
        test_cfg = full_cfg.get("testconfigs", {})
        merged_cfg = {**base_cfg, **test_cfg}
        return TestConfigs(**merged_cfg)
    else:
        return BaseConfigs(**base_cfg)