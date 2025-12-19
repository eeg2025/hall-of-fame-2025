import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

# =============================
# 2) Feature extractor
# =============================

class MetaFeatureExtractor:
    """Builds per-model temporal features from segmentation logits and appends classification outputs."""
    def __init__(self, sfreq=100.0, win_offset=0.5, temps=(0.7, 1.0, 1.3), q_percentiles=(10, 50, 90)):
        self.sfreq = float(sfreq)
        self.dt = 1.0 / self.sfreq
        self.win_offset = float(win_offset)
        self.temps = tuple(temps)
        self.qs = tuple(q_percentiles)

    def _softmax(self, z, t, axis=-1):
        z = z / float(t)
        z = z - np.max(z, axis=axis, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=axis, keepdims=True)

    def _time_grid(self, T):
        return np.arange(T, dtype=np.float32)[None, :] * self.dt

    def _softargmax_time(self, p, tg):
        return np.sum(p * tg, axis=1)

    def _entropy(self, p, eps=1e-12):
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    def _top2_margin(self, x, axis=-1):
        part = np.partition(x, -2, axis=axis)
        top2 = part[..., -2:]
        return top2[..., -1] - top2[..., -2]

    def _quantile_times(self, p, tg):
        cdf = np.cumsum(p, axis=1)
        out = []
        for q in self.qs:
            thr = q / 100.0
            idx = np.argmax(cdf >= thr, axis=1)
            out.append(tg[0, idx])
        return np.stack(out, axis=1)

    def _seg_feats_single(self, Z):  # Z:(N,T)
        N, T = Z.shape
        tg = self._time_grid(T)
        feats = []
        idx = np.argmax(Z, axis=1)
        t_hard = idx * self.dt + self.win_offset
        z_max = np.max(Z, axis=1)
        z_margin = self._top2_margin(Z, axis=1)
        feats += [t_hard[:, None], z_max[:, None], z_margin[:, None]]
        for t in self.temps:
            p = self._softmax(Z, t, axis=1)
            t_rel = self._softargmax_time(p, tg)
            t_abs = t_rel + self.win_offset
            ent = self._entropy(p)
            pmax = np.max(p, axis=1)
            pmargin = self._top2_margin(p, axis=1)
            tvar = np.sum(p * (tg - t_rel[:, None])**2, axis=1)
            qs = self._quantile_times(p, tg)
            feats += [t_abs[:, None], ent[:, None], pmax[:, None], pmargin[:, None], tvar[:, None], qs]
        return np.concatenate(feats, axis=1)  # (N,Fm)

    def _cls_feats_single(self, Y):  # Y:(N,K) or (N,)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            return Y[:, None]
        # also add softmax probs if K>1
        if Y.shape[1] > 1:
            Y_sm = self._softmax(Y, t=1.0, axis=1)
            return np.concatenate([Y, Y_sm], axis=1)
        return Y

    def build_from_logits_store(self, seg_logits_store: dict, cls_outputs_store: dict | None = None):
        """Builds features from dicts: seg[name]->(N,T), cls[name]->(N, K or 1)."""
        Xs = []
        if seg_logits_store:
            for name in seg_logits_store.keys():
                Z = np.asarray(seg_logits_store[name], dtype=np.float32)
                Xs.append(self._seg_feats_single(Z))
        if cls_outputs_store:
            for name in cls_outputs_store.keys():
                Y = np.asarray(cls_outputs_store[name], dtype=np.float32)
                Xs.append(self._cls_feats_single(Y))
        return np.concatenate(Xs, axis=1) if len(Xs) > 1 else Xs[0]

    def build_from_batch(self, seg_logits_list: list[np.ndarray], cls_outputs_list: list[np.ndarray] | None = None):
        """Builds features from lists aligned across models: each seg item is (B,T), each cls item is (B,K or 1)."""
        Xs = [self._seg_feats_single(np.asarray(Z, dtype=np.float32)) for Z in seg_logits_list]
        if cls_outputs_list:
            Xs += [self._cls_feats_single(np.asarray(Y, dtype=np.float32)) for Y in cls_outputs_list]
        return np.concatenate(Xs, axis=1) if len(Xs) > 1 else Xs[0]


# =============================
# 3) Meta regressor
# =============================
class MetaRegressor:
    """Sklearn regressor with OOF-NRMSE for final Ridge model."""
    def __init__(self, kind="ridge", ridge_alphas=None, hgb_params=None, random_state=42):
        self.kind = kind
        self.random_state = random_state
        self.alpha_ = None

        if kind == "ridge":
            self.alphas = ridge_alphas if ridge_alphas is not None else np.logspace(-4, 2, 25)
            self.model = None
        elif kind == "hgb":
            params = dict(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=random_state,
            )
            if hgb_params is not None:
                params.update(hgb_params)
            self.model = HistGradientBoostingRegressor(**params)
        else:
            raise ValueError("Unsupported kind")

    def fit(self, X: np.ndarray, y: np.ndarray, cv: int = 5, report_oof: bool = True) -> None:
        """Fit; for ridge and boosting, prints OOF RMSE/NRMSE of the final pipeline."""
        if self.kind == "ridge":
            # pick best alpha
            ridge_cv = RidgeCV(alphas=self.alphas, scoring="neg_mean_squared_error", cv=cv)
            ridge_cv.fit(X, y)
            self.alpha_ = float(ridge_cv.alpha_)

            # final pipeline with best alpha
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha_, random_state=self.random_state)),
            ])

            # OOF on the final model
            if report_oof:
                kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                y_pred_oof = cross_val_predict(pipe, X, y, cv=kf)
                rmse = np.sqrt(mean_squared_error(y, y_pred_oof))
                nrmse = rmse / np.std(y)
                print(f"OOF RMSE: {rmse:.6f} | OOF NRMSE: {nrmse:.6f}")

            # final fit on full data
            pipe.fit(X, y)
            self.model = pipe
        else:
            # For boosting/hgb also log OOF RMSE/NRMSE if requested
            if report_oof:
                kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                y_pred_oof = cross_val_predict(self.model, X, y, cv=kf)
                rmse = np.sqrt(mean_squared_error(y, y_pred_oof))
                nrmse = rmse / np.std(y)
                print(f"OOF RMSE: {rmse:.6f} | OOF NRMSE: {nrmse:.6f}")

            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        obj = {
            "kind": self.kind,
            "model": self.model,
            "alphas": getattr(self, "alphas", None),
            "alpha_": self.alpha_,
            "random_state": self.random_state,
        }
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path: str) -> "MetaRegressor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        reg = MetaRegressor(
            kind=obj["kind"],
            ridge_alphas=obj.get("alphas"),
            random_state=obj.get("random_state", 42),
        )
        reg.model = obj["model"]
        reg.alpha_ = obj.get("alpha_", None)
        return reg




# =============================
# 1) Meta wrapper (torch.nn.Module)
# =============================

class MetaWrapper(nn.Module):
    """Torch wrapper: runs base models, extracts numpy features, applies sklearn meta-regressor."""
    def __init__(self, seg_models: list[nn.Module] | None,
                 cls_models: list[nn.Module] | None,
                 feature_extractor: MetaFeatureExtractor,
                 meta_regressor: MetaRegressor,
                 use_channels=None,
                 device="cuda"):
        super().__init__()
        self.seg_models = nn.ModuleList(seg_models or [])   # <— вот это важно
        self.cls_models = nn.ModuleList(cls_models or [])   # <— и это
        self.fx = feature_extractor
        self.reg = meta_regressor
        self.use_channels = use_channels
        self.device = device

    @torch.no_grad()
    def _infer_seg_logits(self, x: torch.Tensor):
        Zs = []
        for m in self.seg_models:
            z = m(x).squeeze(1)  # (B,T)
            Zs.append(z.detach().cpu().numpy())
        return Zs  # list of (B,T)

    @torch.no_grad()
    def _infer_cls_outputs(self, x: torch.Tensor):
        Ys = []
        for m in self.cls_models:
            y = m(x)             # (B,K) or (B,1)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.squeeze(-1) if y.ndim == 3 else y
            Ys.append(y.detach().cpu().numpy())
        return Ys  # list of (B,K) or (B,)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Returns (B,1) with predicted absolute time."""
        if self.use_channels is not None:
            x = x[:, self.use_channels, :]
        x = x.to(self.device).float()
        seg_logits = self._infer_seg_logits(x)                     # list of numpy (B,T)
        cls_outputs = self._infer_cls_outputs(x) if self.cls_models else None
        X = self.fx.build_from_batch(seg_logits, cls_outputs)      # numpy (B,F)
        y_hat = self.reg.predict(X).astype(np.float32)             # (B,)
        return torch.from_numpy(y_hat).to(x.device).unsqueeze(1)   # (B,1)
