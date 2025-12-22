#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终提交脚本 (V1-me + V4 融合版)

Challenge 1:
1. (新) 定义 V1 架构 (来自 submission_me.py)
2. (新) 定义 V4 架构 (带 PreProc, 来自 submission.py)
3. (新) 加载 V1 的 43 个模型
4. (新) 加载 V4 的 'select_890_898_random40' 模型
5. (新) 对这两个 *集成* 的预测取平均。
6. (新) 应用 GLOBAL_BIAS 和 0.5s 标签偏移。

Challenge 2:
1. (新) 加载 'weights_challenge_2.pt' (使用 EEGTCNet_Z)。
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys # 用于打印日志

# (新) 从 submission.py 导入 C2 依赖
from braindecode.modules import Chomp1d, MaxNormLinear
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

BASE_PATH = "https://huggingface.co/eeg2025/MIND-CICO/resolve/main/"
# -------------------------------------------------------------------
# (A) 通用组件
# -------------------------------------------------------------------

class SampleChannelNorm(nn.Module):
    """ (V1, V4, C2 通用) """
    def __init__(self, eps: float = 1e-5, center: bool = True, scale: bool = True):
        super().__init__()
        self.eps = eps; self.center = center; self.scale = scale
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.center: x = x - x.mean(dim=-1, keepdim=True)
        if self.scale: x = x / (x.std(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return x

def resolve_path(name="model_file_name"):
    """
    (新) 在根目录中查找文件/文件夹。
    (不再需要 'all_data/')
    """
    root_paths = [
        Path(f"/app/input/res/{name}"),
        Path(f"/app/input/{name}"),
        Path(f"{name}"),
        Path(__file__).parent.joinpath(f"{name}")
    ]
    for p in root_paths:
        if p.exists():
            return str(p)
            
    raise FileNotFoundError(
        f"Could not find {name} in /app/input/res/ or root"
    )

def robust_path(name="model_file_name"):
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
            f"{name} Not Found"
        )
    
class wuaa(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), 1, device=x.device)
# -------------------------------------------------------------------
# (B) C1 - V1 (me) 模型定义 (来自 submission_me.py)
# -------------------------------------------------------------------
class DepthwiseTCN_V1_me(nn.Module):
    """ V1 模型主干网络 (来自 submission_me.py) """
    def __init__(self, C=129, d_model=160, k=15, nblocks=4, dropout=0.1):
        super().__init__()
        self.in_dw = nn.Conv1d(C, C, kernel_size=k, padding=k//2, groups=C)
        self.in_pw = nn.Conv1d(C, d_model, kernel_size=1)
        self.input_norm = SampleChannelNorm(eps=1e-5, center=True, scale=True)
        blocks = []; dil = 1
        for _ in range(nblocks):
            blocks += [nn.Conv1d(d_model, d_model, k, padding=dil*(k//2), dilation=dil, groups=d_model),
                       nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Dropout(dropout)]; dil *= 2
        self.tcn = nn.Sequential(*blocks)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model//4, 1), nn.GELU(), nn.Conv1d(d_model//4, d_model, 1), nn.Sigmoid())
        self.head = nn.Conv1d(d_model, 1, 1)
    def forward(self, x):
        """ (修改) 仅返回 y_hat [0, 2.0) """
        x = self.input_norm(x); z = self.in_pw(self.in_dw(x)); h = self.tcn(z); h = h * self.se(h)
        logits = self.head(h).squeeze(1); P = F.softmax(logits, dim=-1)
        T = logits.size(-1); t_grid = torch.linspace(0.0, 2.0 - 2.0/T, steps=T, device=logits.device)
        y_hat = (P * t_grid).sum(-1)
        return y_hat

class PredictionEnsembleModel_V1_me(nn.Module):
    """ (来自 submission_me.py) """
    def __init__(self, device: torch.device, model_names: list):
        super().__init__()
        self.models = nn.ModuleList()
        print(f"--- [V1-me Ensemble] 正在加载 {len(model_names)} 个 V1 模型... ---", file=sys.stderr)
        for model_name in model_names:
            try:
                # model_path = resolve_path(model_name)
                # data = torch.load(model_path, map_location=device, weights_only=False)
                data = torch.hub.load_state_dict_from_url(BASE_PATH + model_name, map_location=device, weights_only=False)
                hparams = data['hparams']
                model = DepthwiseTCN_V1_me(k=hparams['k'], nblocks=hparams['nblocks'], d_model=hparams['d_model']).to(device)
                model.load_state_dict(data['model'])
                self.models.append(model)
            except Exception as e:
                print(f"警告: 加载 V1-me 模型 {model_name} 失败: {e}", file=sys.stderr)
        if not self.models: raise RuntimeError("错误：未能加载任何一个 V1-me 模型！")
        self.to(device); self.eval()
    def forward(self, x):
        """ (修改) 仅返回 y_hat_adj [0, 2.0) """
        all_y_hats = [model(x) for model in self.models]
        stacked_y_hats = torch.stack(all_y_hats, dim=0)
        avg_y_hat_adj = torch.mean(stacked_y_hats, dim=0)
        return avg_y_hat_adj
    def eval(self):
        super().eval(); [model.eval() for model in self.models]; return self

# -------------------------------------------------------------------
# (C) C1 - V4 (新) 模型定义 (来自 submission.py)
# -------------------------------------------------------------------
class MovingAverage1D(torch.nn.Module):
    def __init__(self, k: int = 5):
        super().__init__(); assert k % 2 == 1; self.k = k
        self.register_buffer("kernel", torch.ones(1, 1, k) / k)
    def forward(self, x):
        B, C, T = x.shape; pad = self.k // 2
        mode = "reflect" if (T > 1 and pad < T) else "replicate"
        x = F.pad(x, (pad, pad), mode=mode)
        w = self.kernel.to(dtype=x.dtype, device=x.device).expand(C, 1, self.k)
        x = F.conv1d(x, w, None, 1, 0, 1, C)
        return x

class EMA1D(torch.nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__(); self.alpha = float(alpha)
    @torch.no_grad()
    def _ema_inplace(self, x):
        alpha = self.alpha; ema = x[..., 0].clone()
        for t in range(1, x.size(-1)):
            ema = alpha * x[..., t] + (1 - alpha) * ema; x[..., t] = ema
        return x
    def forward(self, x):
        state = x.detach().clone(); state = self._ema_inplace(state)
        return state - state.detach() + x

class Detrend1D(torch.nn.Module):
    def __init__(self, k: int = 31):
        super().__init__(); assert k % 2 == 1; self.sma = MovingAverage1D(k)
    def forward(self, x):
        return x - self.sma(x)

def build_preproc(name: str, **kw) -> torch.nn.Module:
    name = (name or "none").lower()
    if name == "none": return torch.nn.Identity()
    if name == "sma": return MovingAverage1D(k=int(kw.get("sma_k", 5)))
    if name == "ema": return EMA1D(alpha=float(kw.get("ema_alpha", 0.2)))
    if name == "detrend": return Detrend1D(k=int(kw.get("detrend_k", 31)))
    raise ValueError(f"Unknown preproc: {name}")

class DepthwiseTCN_V4(nn.Module):
    """ V4 (新) 模型 (带 preproc) """
    def __init__(self, C=129, d_model=160, k=15, nblocks=4, dropout=0.1,
                 preproc_name: str = "none", sma_k: int = 5, ema_alpha: float = 0.2, detrend_k: int = 31):
        super().__init__()
        self.pre = build_preproc(preproc_name, sma_k=sma_k, ema_alpha=ema_alpha, detrend_k=detrend_k)
        self.input_norm = SampleChannelNorm(eps=1e-5, center=True, scale=True)
        self.in_dw = nn.Conv1d(C, C, kernel_size=k, padding=k // 2, groups=C, padding_mode='reflect')
        self.in_pw = nn.Conv1d(C, d_model, kernel_size=1)
        blocks = []; dil = 1
        for _ in range(nblocks):
            blocks += [nn.Conv1d(d_model, d_model, k, padding=dil * (k // 2), dilation=dil, groups=d_model, padding_mode='reflect'),
                       nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Dropout(dropout)]; dil *= 2
        self.tcn = nn.Sequential(*blocks)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1), nn.GELU(), nn.Conv1d(d_model // 4, d_model, 1), nn.Sigmoid())
        self.head = nn.Conv1d(d_model, 1, 1)
    def forward(self, x):
        """ (修改) 仅返回 y_hat [0, 2.0) """
        x = self.pre(x); x = self.input_norm(x)
        z = self.in_pw(self.in_dw(x)); h = self.tcn(z); h = h * self.se(h)
        logits = self.head(h).squeeze(1); P = F.softmax(logits, dim=-1)
        T = logits.size(-1); t_grid = torch.linspace(0.0, 2.0 - 2.0 / T, steps=T, device=logits.device)
        y_hat = (P * t_grid).sum(-1)
        return y_hat # 返回 [0, 2.0)
    
class EnsembleMean_V4(nn.Module):
    """ (来自 submission.py) """
    def __init__(self, device: torch.device, 
                #  weights_dir: Path
                 ):
        super().__init__()
        # if not weights_dir.exists() or not weights_dir.is_dir():
        #     raise RuntimeError(f"权重目录不存在或不是目录: {weights_dir}")
        
        # pt_files = sorted([p for p in weights_dir.iterdir() if p.is_file() and p.suffix == ".pt"])
        pt_files = sorted([
            "t05_t5_s1055_20251101_041145best.pt",
            "t16_t16_s1516_20251101_071936best.pt",
            "t06_t6_s1606_20251101_041841best.pt",
            "t36_t36_s1636_20251101_133902best.pt",
            "t08_t8_s1958_20251101_055700best.pt",
        ])
        # if len(pt_files) == 0:
        #     raise RuntimeError(f"在目录中未找到任何 .pt 文件: {weights_dir}")

        models = []
        print(f"--- [V4 Ensemble] 正在加载 {len(pt_files)} 个 V4 模型... ---", file=sys.stderr)
        for pt in pt_files:
            models.append(self._load_single_model(pt, device))
            
        self.models = nn.ModuleList(models)
        for m in self.models: m.eval(); [p.requires_grad_(False) for p in m.parameters()]
        self.to(device)

    def _load_single_model(self, pt_path: str, device: torch.device) -> nn.Module:
        """ (来自 submission.py) """
        # ckpt = torch.load(str(pt_path), map_location=device, weights_only=False)
        ckpt = torch.hub.load_state_dict_from_url(BASE_PATH + pt_path, map_location=device, weights_only=False)
        if not (isinstance(ckpt, dict) and "hparams" in ckpt and "model" in ckpt):
            raise RuntimeError(f"checkpoint 结构错误")
        hparams = ckpt["hparams"]
        required = ["d_model", "k", "nblocks", "preproc_name", "sma_k", "ema_alpha", "detrend_k"]
        if any(k not in hparams for k in required):
            raise RuntimeError(f"hparams 缺少字段: {required}")
        
        model = DepthwiseTCN_V4(C=129, d_model=int(hparams["d_model"]), k=int(hparams["k"]), 
                              nblocks=int(hparams["nblocks"]), dropout=0.0,
                              preproc_name=hparams["preproc_name"], sma_k=int(hparams["sma_k"]), 
                              ema_alpha=hparams["ema_alpha"], detrend_k=int(hparams["detrend_k"])).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model

    def forward(self, x):
        """ (修改) 仅返回 y_hat_adj [0, 2.0) """
        with torch.no_grad():
            outs = []
            for m in self.models:
                # (修改) 调用 V4 模型的 forward，它只返回 y_hat
                y_hat = m(x) 
                outs.append(y_hat)
            y_mean = torch.stack(outs, dim=0).mean(dim=0) # [B]
        return y_mean

# -------------------------------------------------------------------
# (F) C2 模型定义 (来自 submission.py)
# -------------------------------------------------------------------

class DepthwiseTCN_V1(nn.Module):
    def __init__(self, C=129, d_model=160, k=15, nblocks=4, dropout=0.1):
        super().__init__()
        self.in_dw = nn.Conv1d(C, C, kernel_size=k, padding=k//2, groups=C)
        self.in_pw = nn.Conv1d(C, d_model, kernel_size=1)
        self.input_norm = SampleChannelNorm(eps=1e-5, center=True, scale=True)
        blocks = []; dil = 1
        for _ in range(nblocks):
            blocks += [
                nn.Conv1d(d_model, d_model, k, padding=dil*(k//2), dilation=dil, groups=d_model),
                nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Dropout(dropout),
            ]; dil *= 2
        self.tcn = nn.Sequential(*blocks)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model//4, 1), nn.GELU(), nn.Conv1d(d_model//4, d_model, 1), nn.Sigmoid())
        self.head = nn.Conv1d(d_model, 1, 1)
    
    def forward(self, x):
        x = self.input_norm(x); z = self.in_pw(self.in_dw(x)); h = self.tcn(z); h = h * self.se(h)
        logits = self.head(h).squeeze(1); P = F.softmax(logits, dim=-1)
        T = logits.size(-1); t_grid = torch.linspace(0.0, 2.0 - 2.0/T, steps=T, device=logits.device)
        y_hat = (P * t_grid).sum(-1)
        return y_hat 
    
    def forward_backbone(self, x):
        x = self.input_norm(x)
        z = self.in_pw(self.in_dw(x))
        h = self.tcn(z)
        h = h * self.se(h) 
        return h

class Challenge2Model(nn.Module):
    def __init__(self, c1_backbone: DepthwiseTCN_V1, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = c1_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        d_model = self.backbone.in_pw.out_channels
        self.gap = nn.AdaptiveAvgPool1d(1) 
        self.c2_head = nn.Sequential(
            nn.Linear(d_model, 64), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) 
        )
    def forward(self, x):
        features = self.backbone.forward_backbone(x)
        h_pooled = self.gap(features).squeeze(-1)
        p_factor_pred = self.c2_head(h_pooled) 
        return p_factor_pred -0.05292968079447746
    
class EnsembleMean_C2(nn.Module):
    """
    (!!! 修改版 !!!)
    (新) 加载一个 *目录* 中的 *所有* C2 迁移模型。
    """
    def __init__(self, device: torch.device, 
                #  model_dir: Path
                 ): # <--- (修改) 接受一个 Path，而不是 model_names
        super().__init__()
        self.models = nn.ModuleList()
        
        # if not model_dir.is_dir():
        #     raise RuntimeError(f"C2 权重目录不存在或不是目录: {model_dir}")

        # (!!!) (新) 扫描该目录下的所有 .pt 文件
        # pt_files = sorted([p for p in model_dir.iterdir() if p.is_file() and p.suffix == ".pt"])
        pt_files = sorted([
            'c2_ensemble_from_model_v1_006.pt',
            'c2_ensemble_from_model_v1_011_epoch_01_nrmse_0.990890.pt',
            'c2_ensemble_from_model_v1_013_epoch_02_nrmse_0.991663.pt',
        ])
        # if len(pt_files) == 0:
        #      raise RuntimeError(f"在 C2 目录中未找到任何 .pt 文件: {model_dir}")

        print(f"--- [C2 Ensemble] 正在加载 {len(pt_files)} 个 C2 模型... ---", file=sys.stderr)
        
        for pt_path in pt_files: # <--- (修改) 遍历找到的 Path
            try:
                # (修改) c2_path_str 现在是 pt_path
                # c2_data = torch.load(str(pt_path), map_location=device, weights_only=False)
                c2_data = torch.hub.load_state_dict_from_url(BASE_PATH + pt_path, map_location=device, weights_only=False)
                
                hparams = c2_data['c1_backbone_hparams']
                c1_backbone = DepthwiseTCN_V1(
                    k=hparams['k'], nblocks=hparams['nblocks'], d_model=hparams['d_model']
                ).to(device)
                
                model_c2 = Challenge2Model(c1_backbone, freeze_backbone=True).to(device)
                model_c2.load_state_dict(c2_data['model_state_dict'])
                model_c2.eval()
                self.models.append(model_c2)
            except Exception as e:
                # (修改) 打印 pt_path.name
                print(f"警告: 加载 C2 模型 {pt_path.name} 失败: {e}", file=sys.stderr)
        
        if not self.models:
            raise RuntimeError("错误：未能加载任何一个 C2 集成模型！")
            
        print(f"--- [C2] 成功加载 {len(self.models)} 个 C2 模型。 ---", file=sys.stderr)
        self.to(device); self.eval()

    # ( ... forward 和 eval 方法保持不变 ...)
    def forward(self, x):
        with torch.no_grad():
            outs = []
            for m in self.models:
                y = m(x) # [B, 1]
                outs.append(y)
            y_mean = torch.stack(outs, dim=0).mean(dim=0) # [B, 1]
        return y_mean

    def eval(self):
        super().eval()
        for m in self.models:
            m.eval()
        return self
        
# -------------------------------------------------------------------
# (G) C1 "终极元"集成包装器 (V1-me + V4)
# -------------------------------------------------------------------

class MegaEnsembleModel_V1_V4(nn.Module):
    """
    (新) 加载 V1-me 和 V4 的集成模型，并对它们的预测进行平均。
    """
    def __init__(self, device: torch.device, v1_model_names: list, v4_dir: Path):
        super().__init__()
        
        print("--- [C1 Meta-Ensemble] 正在加载 V1-me 和 V4... ---", file=sys.stderr)
        
        # --- 1. 加载 V1-me (家族 A) ---
        self.ensemble_v1_me = PredictionEnsembleModel_V1_me(
            device=device,
            model_names=v1_model_names
        )
        
        # --- 2. 加载 V4 (家族 B) ---
        self.ensemble_v4 = EnsembleMean_V4(
            device=device,
            # weights_dir=v4_dir
        )
        
        # (!!! 关键 !!!) 在这里粘贴你计算出的偏差值
        GLOBAL_BIAS_VALUE = -0.00383150321431458
        self.register_buffer('GLOBAL_BIAS', 
                             torch.tensor(GLOBAL_BIAS_VALUE, dtype=torch.float32, device=device)) 
        self.NOISE_STD = 0.001
        
        print(f"--- [C1] 成功加载 {len(self.ensemble_v1_me.models)} (V1) + {len(self.ensemble_v4.models)} (V4) 个模型。---", file=sys.stderr)
        self.to(device); self.eval()

    def forward(self, x, submit: bool = True):
        # (新) 统一 forward 接口
        if not submit:
             # 返回 (None, None, y_hat_adj) 以便评估
            y_hat_adj = self._get_avg_y_hat(x)
            return None, None, y_hat_adj
            
        with torch.no_grad():
            avg_y_hat_adj = self._get_avg_y_hat(x) # [B]
            noise = torch.randn_like(avg_y_hat_adj) * self.NOISE_STD
            calibrated_y_hat_adj = avg_y_hat_adj + self.GLOBAL_BIAS+ noise
            final_pred_sec = calibrated_y_hat_adj + 0.5
            final_pred = torch.clamp(final_pred_sec, 0.5, 2.5).unsqueeze(1)
        return final_pred

    def _get_avg_y_hat(self, x):
        """ (新) 辅助函数，计算 V1 和 V4 家族的平均 y_hat """
        # [B]
        y_hat_v1 = self.ensemble_v1_me(x)
        # [B]
        y_hat_v4 = self.ensemble_v4(x)
        
        # (新) 对两个 *集成* 的结果取平均
        avg_y_hat = (y_hat_v1 + y_hat_v4) / 2.0
        return avg_y_hat

    def eval(self):
        super().eval()
        self.ensemble_v1_me.eval()
        self.ensemble_v4.eval()
        return self

# -------------------------------------------------------------------
# (H) 比赛提交类
# -------------------------------------------------------------------

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.C2_DIR_NAME = "c2"
        
        # (新) C1 V1-me 家族的模型
        self.V1_MODEL_NAMES = [f"model_v1_{i:03d}.pt" for i in range(1, 44)]
        
        # (新) C1 V4 家族的文件夹
        self.V4_DIR_NAME = "select_8973_8976_025"

        # (新) C2 训练好的模型路径
        self.C2_MODEL_FILENAME = "c2_ensemble_from_model_v1_006.pt"

    def get_model_challenge_1(self):
        """
        返回 (V1-me + V4) 元集成包装器模型。
        """
        # (新) resolve_path 解析 V4 文件夹
        v4_dir = Path(resolve_path(self.V4_DIR_NAME))
        # (V1_MODEL_NAMES 列表会由 PredictionEnsembleModel_V1_me 内部解析)
        
        model_challenge1 = MegaEnsembleModel_V1_V4(
            device=self.device,
            v1_model_names=self.V1_MODEL_NAMES,
            v4_dir=v4_dir
        )
        
        model_challenge1.eval()
        return model_challenge1


    def get_model_challenge_2(self):
            """
            (!!! 修正版 !!!) 返回 *你* 的 C2 *目录* 集成模型。
            """
            print(f"--- [C2] 正在加载你的 C2 集成... ---", file=sys.stderr)
            try:
                # (!!!) (新) 解析 C2 文件夹路径
                # c2_dir = Path(robust_path(self.C2_DIR_NAME))
                
                model_challenge2 = EnsembleMean_C2(
                    device=self.device,
                    # model_dir=c2_dir  # <--- (修改) 传递 Path
                )
                model_challenge2.eval()
                
            except Exception as e:
                print(f"!!! [C2] 警告: 加载你的 C2 集成失败: {e}", file=sys.stderr)
                print("--- [C2] 返回一个随机初始化的 ZeroModel 作为后备。 ---", file=sys.stderr)
                model_challenge2 = wuaa().to(self.device) # Fallback
                model_challenge2.eval()
                
            return model_challenge2
    

# if __name__ == "__main__":
#     print("--- [本地测试] Mega-Ensemble + C2 迁移模型 开始 ---")
    
#     SFREQ = 100
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"测试设备: {DEVICE}")
    
    
#     try:
#         print("\n正在实例化 Submission 类...")
#         submission = Submission(SFREQ=SFREQ, DEVICE=DEVICE)
#         print("Submission 类实例化成功。")

#         # 3. 测试加载 Challenge 1 模型
#         print("\n正在调用 get_model_challenge_1()...")
#         model_c1 = submission.get_model_challenge_1()
#         if model_c1:
#             print(f"[成功] Challenge 1 (Mega-Ensemble) 加载成功！")
#             # 正确的
#             total_models = len(model_c1.ensemble_v1_me.models) + len(model_c1.ensemble_v4.models)
#             print(f"  - 总共加载了 {total_models} 个子模型。")
            
#             try:
#                 print("  - 尝试使用随机输入进行前向传播...")
#                 dummy_input = torch.rand(2, 129, 200).to(DEVICE)
#                 model_c1.eval() 
#                 with torch.no_grad():
#                     output_c1 = model_c1(dummy_input)
#                 print(f"  - C1 前向传播成功！输出形状: {output_c1.shape}") # 期望 [2, 1]
#             except Exception as e_fwd:
#                 print(f"!!! [错误] C1 模型前向传播失败: {e_fwd}", file=sys.stderr)
#         else:
#             print("!!! [错误] get_model_challenge_1() 返回 None 或失败。", file=sys.stderr)

#         # 4. 测试加载 Challenge 2 模型
#         print("\n正在调用 get_model_challenge_2()...")
#         model_c2 = submission.get_model_challenge_2()
#         if model_c2 and not isinstance(model_c2, wuaa):
#             print("[成功] Challenge 2 (迁移模型) 加载成功！")
#             try:
#                 print("  - 尝试使用随机输入进行前向传播...")
#                 dummy_input = torch.rand(2, 129, 200).to(DEVICE)
#                 model_c2.eval() 
#                 with torch.no_grad():
#                     output_c2 = model_c2(dummy_input)
#                 print(f"  - C2 前向传播成功！输出形状: {output_c2.shape}") # 期望 [2, 1]
#             except Exception as e_fwd:
#                  print(f"!!! [错误] C2 模型前向传播失败: {e_fwd}", file=sys.stderr)
#         elif isinstance(model_c2, wuaa):
#             print("[注意] Challenge 2 加载失败，已回退到 ZeroModel。")
#         else:
#             print("!!! [错误] get_model_challenge_2() 返回 None 或失败。", file=sys.stderr)

#     except Exception as e:
#         print(f"\n!!! [严重错误] 在测试过程中发生异常: {e}", file=sys.stderr)
#         import traceback
#         traceback.print_exc()

#     print("\n--- [本地测试] 结束 ---")

if __name__ == "__main__":
    # Example usage
    s = Submission(SFREQ=100, DEVICE="cpu")
    model_challenge_1 = s.get_model_challenge_1()
    model_challenge_2 = s.get_model_challenge_2()
    print("Models for both challenges are loaded.")