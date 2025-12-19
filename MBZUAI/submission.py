import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

# --- make local package importable ---
_THIS_DIR = Path(__file__).parent
sys.path.append(str(_THIS_DIR))                 
sys.path.append(str(_THIS_DIR / "models"))
sys.path.append(str(_THIS_DIR / "externalizing"))    

# local modules
from models.sneddy_unet import SneddySegUNet1D
from models.attention_sneddy_unet import AttentionSneddyUnet
from models.inception import EEGInceptionSeg1D
from models.factorization_unet import FactorizationSneddyUnet
# from models.submit_wrapper import SubmitWrapper

# externalizing
from externalizing.predictor import ExternalizingPredictor
from externalizing.ridge_model import RidgeModel
# from externalizing.random_forest import RandomForestModel
# from externalizing.boosting_model import BoostingRegressorModel
# from externalizing.knn import KnnModel
from externalizing.feature_extractor import ExternalizingFeaturesExtractor
from externalizing.predictors_blend import ExternalizingPredictorsBlend

# meta
from models.meta_wrapper import MetaWrapper
from models.meta_regressor import MetaRegressor
from models.meta_features import MetaFeatureExtractor


def resolve_path(name="model_file_name"):
    if Path(f"local_submit_output/{name}").exists():
        return f"local_submit_output/{name}"
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory"
        )

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """
        unet_deeper_widen4_v1: 0.212
        unet_deeper_v4: 0.365
        unet_v4: 0.207
        attention_unet_v2: 0.257

        run01: solo unet_deeper_v4
        """

        model_1 = SneddySegUNet1D(
            n_chans=128, n_times=200, sfreq=100,
            c0=64, widen=4, depth_per_stage=5, dropout=0.05, k=15,
            out_channels=1
        ).to(self.device)
        model_1_cpt_path = resolve_path("weights/unet_deeper_widen4_v1.pth")
        model_1.load_state_dict(torch.load(model_1_cpt_path, map_location=self.device))

        model_2 = SneddySegUNet1D(
            n_chans=128, n_times=200, sfreq=100,
            c0=96, widen=2, depth_per_stage=5, dropout=0.05, k=15,
            out_channels=1
        ).to(self.device)
        model_2_cpt_path_new = resolve_path("weights/unet_deeper_v4.pth")
        model_2.load_state_dict(torch.load(model_2_cpt_path_new, map_location=self.device))

        model_3 = SneddySegUNet1D(
            n_chans=128, n_times=200, sfreq=100,
            c0=96, widen=2, depth_per_stage=3, dropout=0.05, k=15,
            out_channels=1
        ).to(self.device)
        model_3_cpt_path = resolve_path("weights/unet_v4.pth")
        model_3.load_state_dict(torch.load(model_3_cpt_path, map_location=self.device))

        model_4 = AttentionSneddyUnet(
            n_chans=128, n_times=200, sfreq=100,
            c0=64, num_stages=5, widen=2.0, depth_per_stage=[2,2,2,1,1],
            bottleneck_type="mhsa", bottleneck_depth=3,
            attn_heads=4, attn_dropout=0.05, ffn_dropout=0.05,
            drop_path=0.1, skip_gating=True
        ).to(self.device)
        model_4_cpt_path = resolve_path("weights/attention_unet_v2.pth")
        model_4.load_state_dict(torch.load(model_4_cpt_path))

        # Inception-style model
        model_5 = EEGInceptionSeg1D(
            n_chans=128, n_times=200, sfreq=100,
            branch_out=32,
            scales_samples_s=(0.5, 0.25, 0.125),
            pooling_sizes=(1, 1),
            dropout=0.12,
            out_channels=1
        ).to(self.device)
        model_5_cpt_path = resolve_path("weights/inception_v0.pth")
        model_5.load_state_dict(torch.load(model_5_cpt_path))

        # Factorization U-Net style model
        model_6 = FactorizationSneddyUnet(
            n_chans=128, n_times=200, sfreq=100, c0=96, widen=2,
            n_stages=4,
            depth_per_stage=2, dropout=0.1, k=7, out_channels=1,
            fm_factors_front=64, fm_dropout_front=0.05,
            use_stage_fm=True,
            fm_factors_stage=32, fm_dropout_stage=0.05
        ).to(self.device)
        model_6_cpt_path = resolve_path("weights/factorization_unet_v1_finetune.pth")
        model_6.load_state_dict(torch.load(model_6_cpt_path))
        
        # stacking
        seg_models = [model_1, model_2, model_3, model_4, model_5, model_6] 
        reg = MetaRegressor.load(resolve_path("weights/meta_hgb_new.pkl"))
        fx_hgbr = MetaFeatureExtractor(sfreq=100, win_offset=0.5, temps=(0.5, 0.7, 0.8, 1.))
        meta_hgbr = MetaWrapper(seg_models=seg_models, cls_models=[],
                   feature_extractor=fx_hgbr, meta_regressor=reg,
                   use_channels=np.arange(128), device=self.device).to(self.device)
        return meta_hgbr


    def get_model_challenge_2(self):
        usecols  = [
            'lag10_corr_diag_mean',
            'lag10_corr_entropy',
            'lag25_corr_entropy',
            'lag100_A_diag_mean',
            'lag25_corr_diag_mean',
            'lag5_corr_diag_mean',
            'lag25_corr_mean_abs',
            'lag50_corr_mean_abs',
            'lag100_corr_entropy',
            'lag50_A_diag_mean',
            'lag5_corr_off_mean',
            'lag5_corr_entropy',
            'lag100_A_mean_abs',
            'lag50_corr_entropy',
            'lag100_A_asym_mean_abs',
            'lag100_A_fro',
            'lag50_corr_off_mean',
            'lag100_A_sparsity@0.05',
            'lag10_corr_mean_abs',
            'lag50_A_mean_abs',
            'lag5_A_off_mean',
            'lag25_A_mean_abs',
            'lag10_A_off_mean'
        ]
        ridge_model = RidgeModel(usecols)
        ridge_cpt_path = resolve_path("weights/short_ridge.pt")
        ridge_model.load_state_dict(torch.load(ridge_cpt_path)['state_dict'])

        fe = ExternalizingFeaturesExtractor(lags=(5, 10, 25, 50, 100))

        model_challenge2 = ExternalizingPredictorsBlend(
            feature_extractor=fe, ridge_models=[ridge_model], 
            ridge_weights=(1,),
            history_size=100, history_weight=0.75,
            clip_min=-0.45, clip_max=0.35
        )
        return model_challenge2
