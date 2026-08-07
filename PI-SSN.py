# -*- coding: utf-8 -*-
# NOTE: The authors are continuously optimizing and updating the PI-SSN model.
# 注：作者正在持续优化和更新 PI-SSN 模型，后续版本可能进一步完善。

import copy
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 0. Constants and reproducibility
# ============================================================================

REQUIRED_BANDS = [
    405, 430, 450, 550, 560,
    570, 650, 685, 710, 850
]

FEATURE_NAMES = [
    "Bio_F1",
    "Bio_F2",
    "Bio_F3",
    "Fib_F1",
    "Fib_F2",
    "Fib_F3",
    "Fib_F4",
    "Fib_F5",
    "Fib_F6",
]

TARGET_NAMES = ["N", "CP", "ADF", "NDF"]


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Improve reproducibility for CUDA.
    # Deterministic algorithms may be slower than non-deterministic algorithms.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# 1. Feature extraction based on Table 2 of the paper
# ============================================================================

def _standardize_band_column_names(df_bands: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wavelength-like column names to integer wavelengths when possible.

    Examples
    --------
    "405"     -> 405
    "405.0"   -> 405
    405.0     -> 405

    Other column names are kept unchanged.
    """
    df_bands = df_bands.copy()
    new_columns = []

    for column in df_bands.columns:
        try:
            wavelength_float = float(column)

            if wavelength_float.is_integer():
                new_columns.append(int(wavelength_float))
            else:
                new_columns.append(column)

        except (TypeError, ValueError):
            new_columns.append(column)

    df_bands.columns = new_columns
    return df_bands


def _validate_band_dataframe(df_bands: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the reflectance DataFrame.
    """
    if not isinstance(df_bands, pd.DataFrame):
        raise TypeError(
            "df_bands must be a pandas.DataFrame."
        )

    df_bands = _standardize_band_column_names(df_bands)

    missing_bands = [
        band for band in REQUIRED_BANDS
        if band not in df_bands.columns
    ]

    if missing_bands:
        raise ValueError(
            "The following required wavelength columns are missing: "
            f"{missing_bands}"
        )

    # Keep only the required bands and force numeric values.
    df_bands = df_bands.loc[:, REQUIRED_BANDS].copy()

    for band in REQUIRED_BANDS:
        df_bands[band] = pd.to_numeric(
            df_bands[band],
            errors="coerce"
        )

    return df_bands


def extract_orthogonal_features(
    df_bands: pd.DataFrame,
    invalid_fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Compute the nine orthogonal spectral features listed in Table 2.

    Parameters
    ----------
    df_bands : pandas.DataFrame
        Reflectance values for the 10 AMS-10 bands:

        405, 430, 450, 550, 560,
        570, 650, 685, 710 and 850 nm.

        Rows represent samples and columns represent wavelengths.

    invalid_fill_value : float, default=0.0
        Value used to replace undefined feature values caused by invalid
        reflectance values or zero denominators.

    Returns
    -------
    pandas.DataFrame
        Nine spectral features:

        Bio_F1, Bio_F2, Bio_F3,
        Fib_F1, Fib_F2, Fib_F3,
        Fib_F4, Fib_F5, Fib_F6.

    Notes
    -----
    OR means original reflectance space.

    Log means:

        log(1 / R)

    Reflectance must be positive for logarithmic transformation.
    Non-positive reflectance is treated as invalid and replaced after
    feature calculation.
    """
    df_bands = _validate_band_dataframe(df_bands)

    # Logarithmic transformation based on the Beer-Lambert law.
    # Non-positive reflectance cannot be transformed logarithmically.
    positive_bands = df_bands.where(df_bands > 0)
    df_log = np.log(1.0 / positive_bands)

    feats = pd.DataFrame(
        index=df_bands.index,
        dtype=np.float64
    )

    # ------------------------------------------------------------------------
    # Biochemical traits: N and CP
    # ------------------------------------------------------------------------

    # 3B-Pigment (560, 710, 405) [OR]
    feats["Bio_F1"] = (
        (1.0 / df_bands[560] - 1.0 / df_bands[710])
        * df_bands[405]
    )

    # 4B-DD (405, 570, 560, 710) [OR]
    feats["Bio_F2"] = (
        (df_bands[405] - df_bands[570])
        / (df_bands[560] - df_bands[710])
    )

    # 4B-DD (405, 550, 570, 650) [Log]
    feats["Bio_F3"] = (
        (df_log[405] - df_log[550])
        / (df_log[570] - df_log[650])
    )

    # ------------------------------------------------------------------------
    # Structural traits: ADF and NDF
    # ------------------------------------------------------------------------

    # 4B-DD (430, 560, 450, 570) [OR]
    feats["Fib_F1"] = (
        (df_bands[430] - df_bands[560])
        / (df_bands[450] - df_bands[570])
    )

    # 3B-VARI (405, 550, 850) [Log]
    feats["Fib_F2"] = (
        (df_log[405] - df_log[550])
        / (df_log[405] + df_log[550] - df_log[850])
    )

    # 4B-DD (405, 560, 450, 850) [Log]
    feats["Fib_F3"] = (
        (df_log[405] - df_log[560])
        / (df_log[450] - df_log[850])
    )

    # 4B-DD (570, 710, 405, 550) [OR]
    feats["Fib_F4"] = (
        (df_bands[570] - df_bands[710])
        / (df_bands[405] - df_bands[550])
    )

    # 3B-MTCI (550, 710, 405) [OR]
    feats["Fib_F5"] = (
        (df_bands[550] - df_bands[710])
        / (df_bands[710] - df_bands[405])
    )

    # 4B-DD (450, 850, 405, 550) [Log]
    feats["Fib_F6"] = (
        (df_log[450] - df_log[850])
        / (df_log[405] - df_log[550])
    )

    # Replace undefined numerical results.
    feats = feats.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feats = feats.fillna(invalid_fill_value)

    return feats.loc[:, FEATURE_NAMES]


def build_temporal_features(
    df_bands_list: Sequence[pd.DataFrame],
    time_points: Sequence[str]
) -> pd.DataFrame:
    """
    Extract and concatenate features from multiple growth stages.

    Parameters
    ----------
    df_bands_list : sequence of pandas.DataFrame
        One reflectance DataFrame for each growth stage.

    time_points : sequence of str
        Growth-stage names, for example:

            ["T1", "T2", "T3"]

    Returns
    -------
    pandas.DataFrame
        Concatenated temporal features. For three stages, the output contains
        27 variables in the following stage-block order:

            T1 features, T2 features, T3 features.

    Notes
    -----
    All DataFrames must use compatible sample indices. The function aligns
    samples by their index through pandas concatenation.
    """
    if len(df_bands_list) == 0:
        raise ValueError(
            "df_bands_list must contain at least one growth stage."
        )

    if len(df_bands_list) != len(time_points):
        raise ValueError(
            "df_bands_list and time_points must have the same length."
        )

    if len(set(time_points)) != len(time_points):
        raise ValueError(
            "Each value in time_points must be unique."
        )

    all_features = []

    for df_bands, time_point in zip(
        df_bands_list,
        time_points
    ):
        stage_features = extract_orthogonal_features(df_bands)
        stage_features = stage_features.add_suffix(
            f"_{time_point}"
        )
        all_features.append(stage_features)

    temporal_features = pd.concat(
        all_features,
        axis=1,
        join="inner"
    )

    return temporal_features


# ============================================================================
# 2. Trait-driven decoupled temporal attention
# ============================================================================

class TemporalAttention(nn.Module):
    """
    Feed-forward temporal attention module.

    Given:

        X_time in R^(batch × time × feature)

    this module calculates:

        e_t = tanh(W_a X_t + b_a)

        alpha_t = softmax(w_e^T e_t)

        C = sum_t alpha_t X_t
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 8
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of features at each time point.

        hidden_dim : int
            Dimension of the intermediate attention representation.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be a positive integer."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be a positive integer."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Corresponds to W_a X_t + b_a.
        self.hidden_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Corresponds to w_e^T e_t.
        self.score_projection = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:

                (batch_size, n_time, input_dim)

        Returns
        -------
        context : torch.Tensor
            Temporally aggregated context vector with shape:

                (batch_size, input_dim)

        attention_weights : torch.Tensor
            Normalized temporal weights with shape:

                (batch_size, n_time, 1)
        """
        if x.ndim != 3:
            raise ValueError(
                "TemporalAttention expects a 3D tensor with shape "
                "(batch_size, n_time, input_dim)."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features per time point, "
                f"but received {x.shape[-1]}."
            )

        # Equation (8)
        attention_hidden = torch.tanh(
            self.hidden_projection(x)
        )

        # Scalar score for each time point.
        attention_scores = self.score_projection(
            attention_hidden
        )

        # Equation (9): normalize across the temporal dimension.
        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # Equation (10)
        context = torch.sum(
            attention_weights * x,
            dim=1
        )

        return context, attention_weights


# ============================================================================
# 3. PI-SSN model
# ============================================================================

class PI_SSN(nn.Module):
    """
    Physics-Informed Sparse Shallow Network.

    Architecture
    ------------
    1. Biochemical temporal attention branch:
       N and CP.

    2. Structural temporal attention branch:
       ADF and NDF.

    3. Smooth inference stream:
       Single hidden layer with Tanh activation.

    4. Sparse feature stream:
       Single hidden layer with LeakyReLU activation and L1 regularization.

    Unlike a residual or skip-connected architecture, each prediction stream
    receives only its trait-specific temporal context vector, consistent with
    the PI-SSN framework illustrated in the paper.
    """

    def __init__(
        self,
        n_time: int = 3,
        feat_per_time: int = 9,
        attention_hidden_dim: int = 8,
        bio_hidden_dim: int = 16,
        struc_hidden_dim: int = 12,
        n_bio: int = 2,
        n_struc: int = 2,
        leaky_relu_slope: float = 0.1
    ) -> None:
        """
        Parameters
        ----------
        n_time : int
            Number of temporal observations.

        feat_per_time : int
            Number of features at each time point.

        attention_hidden_dim : int
            Hidden dimension of each temporal attention module.

        bio_hidden_dim : int
            Hidden neurons in the biochemical stream.

        struc_hidden_dim : int
            Hidden neurons in the structural stream.

        n_bio : int
            Number of biochemical outputs. The paper uses 2: N and CP.

        n_struc : int
            Number of structural outputs. The paper uses 2: ADF and NDF.

        leaky_relu_slope : float
            Negative slope for LeakyReLU.
        """
        super().__init__()

        if n_time <= 0:
            raise ValueError(
                "n_time must be a positive integer."
            )

        if feat_per_time <= 0:
            raise ValueError(
                "feat_per_time must be a positive integer."
            )

        self.n_time = n_time
        self.feat_per_time = feat_per_time
        self.n_bio = n_bio
        self.n_struc = n_struc

        # Independent attention modules for different trait groups.
        self.attn_bio = TemporalAttention(
            input_dim=feat_per_time,
            hidden_dim=attention_hidden_dim
        )

        self.attn_struc = TemporalAttention(
            input_dim=feat_per_time,
            hidden_dim=attention_hidden_dim
        )

        # Smooth inference stream for N and CP.
        self.bio_stream = nn.Sequential(
            nn.Linear(
                feat_per_time,
                bio_hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                bio_hidden_dim,
                n_bio
            )
        )

        # Sparse feature stream for ADF and NDF.
        self.struc_stream = nn.Sequential(
            nn.Linear(
                feat_per_time,
                struc_hidden_dim
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_slope
            ),
            nn.Linear(
                struc_hidden_dim,
                n_struc
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the network parameters.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_flat: torch.Tensor,
        return_attn: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Forward propagation.

        Parameters
        ----------
        x_flat : torch.Tensor
            Flattened multi-temporal input with shape:

                (batch_size, n_time * feat_per_time)

            For the paper configuration:

                (batch_size, 27)

        return_attn : bool
            If True, return biochemical and structural attention weights.

        Returns
        -------
        predictions : torch.Tensor
            Predictions in the order:

                [N, CP, ADF, NDF]

        attn_bio : torch.Tensor, optional
            Biochemical temporal attention weights.

        attn_struc : torch.Tensor, optional
            Structural temporal attention weights.
        """
        if x_flat.ndim != 2:
            raise ValueError(
                "PI_SSN expects a 2D flattened input tensor with shape "
                "(batch_size, n_time * feat_per_time)."
            )

        expected_features = (
            self.n_time * self.feat_per_time
        )

        if x_flat.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} input features "
                f"({self.n_time} time points × "
                f"{self.feat_per_time} features), "
                f"but received {x_flat.shape[1]}."
            )

        batch_size = x_flat.shape[0]

        x_time = x_flat.reshape(
            batch_size,
            self.n_time,
            self.feat_per_time
        )

        # Trait-specific temporal context vectors.
        context_bio, attn_bio = self.attn_bio(x_time)
        context_struc, attn_struc = self.attn_struc(x_time)

        # Each stream receives its corresponding attention context.
        bio_predictions = self.bio_stream(context_bio)
        struc_predictions = self.struc_stream(context_struc)

        predictions = torch.cat(
            [bio_predictions, struc_predictions],
            dim=1
        )

        if return_attn:
            return predictions, attn_bio, attn_struc

        return predictions


# ============================================================================
# 4. Biology-constrained joint loss
# ============================================================================

def _batch_covariance(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Calculate batch covariance using the definition in Equation (14).

    The denominator is batch size B rather than B - 1.
    """
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)

    return torch.mean(
        x_centered * y_centered
    )


def structural_l1_penalty(
    model: PI_SSN
) -> torch.Tensor:
    """
    Calculate the structural-stream L1 sparsity penalty.

    Following the feature-selection interpretation in the paper, L1 is applied
    only to the first fully connected weight matrix of the structural stream.
    Bias terms are not penalized.
    """
    first_structural_layer = model.struc_stream[0]

    if not isinstance(first_structural_layer, nn.Linear):
        raise TypeError(
            "The first module of struc_stream must be nn.Linear."
        )

    return torch.sum(
        torch.abs(first_structural_layer.weight)
    )


def physics_informed_loss(
    model: PI_SSN,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lambda_phy: float = 0.5,
    lambda_l1: float = 0.01,
    gamma_struc: float = 1.0,
    return_components: bool = False
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Dict[str, torch.Tensor]]
]:
    """
    Calculate the PI-SSN biology-constrained joint loss.

    The total loss follows:

        L_total = L_reg
                  + lambda_phy * Psi_phy
                  + lambda_l1 * L_L1

    Heterogeneous regression loss
    -----------------------------
    Biochemical traits:

        N, CP -> MSE

    Structural traits:

        ADF, NDF -> gamma_struc * SmoothL1

    Biological trade-off constraint
    --------------------------------
    All biochemical-structural pairs are included:

        N   vs. ADF
        N   vs. NDF
        CP  vs. ADF
        CP  vs. NDF

    The physical penalty follows Equation (15):

        Psi_phy =
            sum_i sum_j Softplus(Cov(y_i, y_j))

    Parameters
    ----------
    model : PI_SSN
        PI-SSN model.

    predictions : torch.Tensor
        Shape (batch_size, 4), ordered as:

            [N, CP, ADF, NDF]

    targets : torch.Tensor
        Ground-truth values with the same shape and order.

    lambda_phy : float
        Weight of the biological trade-off constraint.

    lambda_l1 : float
        Weight of the structural L1 sparsity penalty.

    gamma_struc : float
        Weight assigned to the structural Smooth L1 loss.

    return_components : bool
        If True, also return individual loss components.

    Returns
    -------
    total_loss : torch.Tensor
        Scalar total loss.

    components : dict, optional
        Individual detached loss components.
    """
    if predictions.ndim != 2 or predictions.shape[1] != 4:
        raise ValueError(
            "predictions must have shape (batch_size, 4) "
            "in the order [N, CP, ADF, NDF]."
        )

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have identical shapes."
        )

    if lambda_phy < 0:
        raise ValueError(
            "lambda_phy must be non-negative."
        )

    if lambda_l1 < 0:
        raise ValueError(
            "lambda_l1 must be non-negative."
        )

    if gamma_struc < 0:
        raise ValueError(
            "gamma_struc must be non-negative."
        )

    # ------------------------------------------------------------------------
    # 1. Heterogeneous regression loss, Equation (13)
    # ------------------------------------------------------------------------

    loss_n = F.mse_loss(
        predictions[:, 0],
        targets[:, 0]
    )

    loss_cp = F.mse_loss(
        predictions[:, 1],
        targets[:, 1]
    )

    loss_adf = F.smooth_l1_loss(
        predictions[:, 2],
        targets[:, 2]
    )

    loss_ndf = F.smooth_l1_loss(
        predictions[:, 3],
        targets[:, 3]
    )

    regression_loss = (
        loss_n
        + loss_cp
        + gamma_struc * (loss_adf + loss_ndf)
    )

    # ------------------------------------------------------------------------
    # 2. Biological trade-off constraint, Equations (14)-(15)
    # ------------------------------------------------------------------------

    biochemical_indices = [0, 1]   # N and CP
    structural_indices = [2, 3]    # ADF and NDF

    physics_penalty = predictions.new_tensor(0.0)

    covariance_terms = {}

    for bio_index in biochemical_indices:
        for struc_index in structural_indices:
            covariance = _batch_covariance(
                predictions[:, bio_index],
                predictions[:, struc_index]
            )

            pair_name = (
                f"{TARGET_NAMES[bio_index]}_"
                f"{TARGET_NAMES[struc_index]}"
            )

            covariance_terms[pair_name] = covariance

            # Strictly follows Equation (15) in the paper.
            physics_penalty = (
                physics_penalty
                + F.softplus(covariance)
            )

    # ------------------------------------------------------------------------
    # 3. Structural L1 sparsity, Equation (11)
    # ------------------------------------------------------------------------

    l1_penalty = structural_l1_penalty(model)

    # ------------------------------------------------------------------------
    # 4. Total loss, Equation (12)
    # ------------------------------------------------------------------------

    total_loss = (
        regression_loss
        + lambda_phy * physics_penalty
        + lambda_l1 * l1_penalty
    )

    if not return_components:
        return total_loss

    components = {
        "total": total_loss.detach(),
        "regression": regression_loss.detach(),
        "loss_N": loss_n.detach(),
        "loss_CP": loss_cp.detach(),
        "loss_ADF": loss_adf.detach(),
        "loss_NDF": loss_ndf.detach(),
        "physics": physics_penalty.detach(),
        "l1": l1_penalty.detach(),
    }

    for pair_name, covariance in covariance_terms.items():
        components[f"cov_{pair_name}"] = covariance.detach()

    return total_loss, components


# ============================================================================
# 5. Training utilities
# ============================================================================

def _validate_training_arrays(
    X: np.ndarray,
    y: np.ndarray,
    n_time: int,
    feat_per_time: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate training arrays and convert them to float32.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array."
        )

    if y.ndim != 2:
        raise ValueError(
            "y must be a 2D array."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    expected_features = n_time * feat_per_time

    if X.shape[1] != expected_features:
        raise ValueError(
            f"X must contain {expected_features} columns "
            f"({n_time} time points × {feat_per_time} features), "
            f"but received {X.shape[1]} columns."
        )

    if y.shape[1] != 4:
        raise ValueError(
            "y must contain four target columns ordered as "
            "[N, CP, ADF, NDF]."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )

    return X, y


def _evaluate_loss(
    model: PI_SSN,
    X: torch.Tensor,
    y: torch.Tensor,
    lambda_phy: float,
    lambda_l1: float,
    gamma_struc: float
) -> float:
    """
    Evaluate the complete PI-SSN loss on a full dataset.
    """
    model.eval()

    with torch.no_grad():
        predictions = model(X)

        loss = physics_informed_loss(
            model=model,
            predictions=predictions,
            targets=y,
            lambda_phy=lambda_phy,
            lambda_l1=lambda_l1,
            gamma_struc=gamma_struc
        )

    return float(loss.item())


def train_pi_ssn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    n_time: int = 3,
    feat_per_time: int = 9,
    epochs: int = 600,
    lr: float = 0.008,
    batch_size: int = 32,
    validation_fraction: float = 0.20,
    lambda_phy: float = 0.5,
    lambda_l1: float = 0.01,
    gamma_struc: float = 1.0,
    scheduler_patience: int = 20,
    early_stopping_patience: int = 60,
    min_delta: float = 1e-6,
    random_state: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True
) -> Tuple[
    PI_SSN,
    np.ndarray,
    Optional[np.ndarray],
    Dict[str, List[float]]
]:
    """
    Train the PI-SSN model.

    The supplied training set is internally divided into:

        optimization subset + validation subset

    The independent test set is never used for parameter updating,
    learning-rate scheduling, checkpoint selection or early stopping.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features with shape:

            (n_samples, n_time * feat_per_time)

    y_train : numpy.ndarray
        Training targets with shape:

            (n_samples, 4)

        Target order:

            [N, CP, ADF, NDF]

    X_test : numpy.ndarray or None
        Independent test features.

    y_test : numpy.ndarray or None
        Accepted for compatibility and optional external evaluation.
        It is not used for model selection or training.

    n_time : int
        Number of time points. Use:

            n_time=3 for TS mode
            n_time=1 for Inst mode

    feat_per_time : int
        Number of features at each time point.

    epochs : int
        Maximum number of training epochs.

    lr : float
        Initial learning rate.

    batch_size : int
        Training batch size.

    validation_fraction : float
        Fraction of X_train used for internal validation.

    lambda_phy : float
        Weight of the biological covariance constraint.

    lambda_l1 : float
        Weight of the structural L1 penalty.

    gamma_struc : float
        Weight of structural Smooth L1 losses.

    scheduler_patience : int
        Patience for ReduceLROnPlateau.

    early_stopping_patience : int
        Number of epochs without validation improvement before stopping.

    min_delta : float
        Minimum validation loss improvement.

    random_state : int
        Random seed.

    device : str, torch.device or None
        Training device. If None, CUDA is used when available.

    verbose : bool
        Print training progress.

    Returns
    -------
    model : PI_SSN
        Trained model with the best validation weights loaded.

    pred_train : numpy.ndarray
        Predictions for all samples supplied in X_train.

    pred_test : numpy.ndarray or None
        Predictions for X_test.

    history : dict
        Training history containing:

            train_loss
            validation_loss
            learning_rate

    Notes
    -----
    Predictions remain in the same target scale as y_train. If y_train has
    been standardized, inverse transformation must be performed outside this
    function.
    """
    del y_test  # Never used in model training or checkpoint selection.

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    set_random_seed(random_state)

    X_train, y_train = _validate_training_arrays(
        X_train,
        y_train,
        n_time=n_time,
        feat_per_time=feat_per_time
    )

    if X_test is not None:
        X_test = np.asarray(
            X_test,
            dtype=np.float32
        )

        if X_test.ndim != 2:
            raise ValueError(
                "X_test must be a 2D array."
            )

        expected_features = n_time * feat_per_time

        if X_test.shape[1] != expected_features:
            raise ValueError(
                f"X_test must contain {expected_features} columns."
            )

        if not np.isfinite(X_test).all():
            raise ValueError(
                "X_test contains NaN or infinite values."
            )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    # Internal validation split. The independent test set remains untouched.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True
    )

    X_fit_tensor = torch.from_numpy(X_fit)
    y_fit_tensor = torch.from_numpy(y_fit)

    X_val_tensor = torch.from_numpy(X_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    X_train_tensor = torch.from_numpy(X_train).to(device)

    if X_test is not None:
        X_test_tensor = torch.from_numpy(X_test).to(device)
    else:
        X_test_tensor = None

    training_dataset = TensorDataset(
        X_fit_tensor,
        y_fit_tensor
    )

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(random_state)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=data_loader_generator
    )

    model = PI_SSN(
        n_time=n_time,
        feat_per_time=feat_per_time
    ).to(device)

    # Adam is used without additional L2 weight decay because the paper
    # explicitly defines L1 sparsity for the structural stream.
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
        min_lr=1e-6
    )

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": []
    }

    if verbose:
        print("=" * 72)
        print("Training PI-SSN")
        print(f"Device: {device}")
        print(f"Training samples: {len(X_fit)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Input mode: {n_time} time point(s)")
        print(
            f"Input dimension: "
            f"{n_time} × {feat_per_time} = "
            f"{n_time * feat_per_time}"
        )
        print("=" * 72)

    for epoch in range(1, epochs + 1):
        model.train()

        accumulated_loss = 0.0
        accumulated_samples = 0

        for batch_X, batch_y in training_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(batch_X)

            loss = physics_informed_loss(
                model=model,
                predictions=predictions,
                targets=batch_y,
                lambda_phy=lambda_phy,
                lambda_l1=lambda_l1,
                gamma_struc=gamma_struc
            )

            loss.backward()
            optimizer.step()

            current_batch_size = batch_X.shape[0]

            accumulated_loss += (
                loss.item() * current_batch_size
            )
            accumulated_samples += current_batch_size

        mean_training_loss = (
            accumulated_loss / accumulated_samples
        )

        validation_loss = _evaluate_loss(
            model=model,
            X=X_val_tensor,
            y=y_val_tensor,
            lambda_phy=lambda_phy,
            lambda_l1=lambda_l1,
            gamma_struc=gamma_struc
        )

        scheduler.step(validation_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(
            mean_training_loss
        )
        history["validation_loss"].append(
            validation_loss
        )
        history["learning_rate"].append(
            current_lr
        )

        improved = (
            validation_loss
            < best_validation_loss - min_delta
        )

        if improved:
            best_validation_loss = validation_loss

            # Deep copy is required to preserve the actual best checkpoint.
            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (
            epoch == 1
            or epoch % 20 == 0
            or improved
            or epoch == epochs
        ):
            marker = " *" if improved else ""

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train loss: {mean_training_loss:.6f} | "
                f"Val loss: {validation_loss:.6f} | "
                f"LR: {current_lr:.6g}{marker}"
            )

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):
            if verbose:
                print(
                    "Early stopping: validation loss did not improve "
                    f"for {early_stopping_patience} epochs."
                )
            break

    # Restore the checkpoint with the lowest internal validation loss.
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_train = (
            model(X_train_tensor)
            .cpu()
            .numpy()
        )

        if X_test_tensor is not None:
            pred_test = (
                model(X_test_tensor)
                .cpu()
                .numpy()
            )
        else:
            pred_test = None

    if verbose:
        print("=" * 72)
        print(
            f"Training completed. "
            f"Best validation loss: {best_validation_loss:.6f}"
        )
        print("=" * 72)

    return model, pred_train, pred_test, history


# ============================================================================
# 6. Attention extraction and evaluation
# ============================================================================

def extract_attention_weights(
    model: PI_SSN,
    X: np.ndarray,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract temporal attention weights from a trained PI-SSN model.

    Parameters
    ----------
    model : PI_SSN
        Trained model.

    X : numpy.ndarray
        Input features with shape:

            (n_samples, n_time * feat_per_time)

    device : str, torch.device or None
        Inference device.

    Returns
    -------
    biochemical_attention : numpy.ndarray
        Shape:

            (n_samples, n_time)

    structural_attention : numpy.ndarray
        Shape:

            (n_samples, n_time)
    """
    X = np.asarray(X, dtype=np.float32)

    expected_features = (
        model.n_time * model.feat_per_time
    )

    if X.ndim != 2 or X.shape[1] != expected_features:
        raise ValueError(
            f"X must have shape (n_samples, {expected_features})."
        )

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    X_tensor = torch.from_numpy(X).to(device)

    model.eval()

    with torch.no_grad():
        _, attn_bio, attn_struc = model(
            X_tensor,
            return_attn=True
        )

    biochemical_attention = (
        attn_bio.squeeze(-1)
        .cpu()
        .numpy()
    )

    structural_attention = (
        attn_struc.squeeze(-1)
        .cpu()
        .numpy()
    )

    return biochemical_attention, structural_attention


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculate R², RMSE, MAE and RPD for all four traits.

    RPD is calculated as:

        standard deviation of measured values / RMSE
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.ndim != 2 or y_true.shape[1] != 4:
        raise ValueError(
            "Expected arrays with shape (n_samples, 4)."
        )

    metrics = {}

    for index, target_name in enumerate(TARGET_NAMES):
        observed = y_true[:, index]
        predicted = y_pred[:, index]

        rmse = np.sqrt(
            mean_squared_error(
                observed,
                predicted
            )
        )

        standard_deviation = np.std(
            observed,
            ddof=1
        )

        if rmse > 0:
            rpd = standard_deviation / rmse
        else:
            rpd = np.inf

        metrics[target_name] = {
            "R2": r2_score(
                observed,
                predicted
            ),
            "RMSE": rmse,
            "MAE": mean_absolute_error(
                observed,
                predicted
            ),
            "RPD": rpd
        }

    return metrics


def print_regression_metrics(
    metrics: Dict[str, Dict[str, float]]
) -> None:
    """
    Print regression metrics in a readable table.
    """
    print(
        f"{'Target':<8}"
        f"{'R²':>10}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'RPD':>12}"
    )
    print("-" * 54)

    for target_name in TARGET_NAMES:
        target_metrics = metrics[target_name]

        print(
            f"{target_name:<8}"
            f"{target_metrics['R2']:>10.3f}"
            f"{target_metrics['RMSE']:>12.3f}"
            f"{target_metrics['MAE']:>12.3f}"
            f"{target_metrics['RPD']:>12.3f}"
        )


# ============================================================================
# 7. Model checkpoint utilities
# ============================================================================

def save_pi_ssn_checkpoint(
    model: PI_SSN,
    file_path: str,
    extra_information: Optional[Dict] = None
) -> None:
    """
    Save model weights and architecture configuration.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_time": model.n_time,
            "feat_per_time": model.feat_per_time,
            "n_bio": model.n_bio,
            "n_struc": model.n_struc
        }
    }

    if extra_information is not None:
        checkpoint["extra_information"] = extra_information

    torch.save(checkpoint, file_path)


def load_pi_ssn_checkpoint(
    file_path: str,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[PI_SSN, Dict]:
    """
    Load a saved PI-SSN checkpoint.

    Returns
    -------
    model : PI_SSN
        Loaded model.

    checkpoint : dict
        Complete checkpoint information.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    checkpoint = torch.load(
        file_path,
        map_location=device
    )

    model_config = checkpoint["model_config"]

    model = PI_SSN(
        n_time=model_config["n_time"],
        feat_per_time=model_config["feat_per_time"],
        n_bio=model_config.get("n_bio", 2),
        n_struc=model_config.get("n_struc", 2)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================================
# 8. Example usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PI-SSN example")
    print("Please replace the synthetic data with your own UAV data.")
    print("=" * 72)

    set_random_seed(42)

    # ------------------------------------------------------------------------
    # Example data structure
    # ------------------------------------------------------------------------
    #
    # Real-data example:
    #
    # df_T1 = pd.read_excel("stage_T1_reflectance.xlsx", index_col=0)
    # df_T2 = pd.read_excel("stage_T2_reflectance.xlsx", index_col=0)
    # df_T3 = pd.read_excel("stage_T3_reflectance.xlsx", index_col=0)
    #
    # X_dataframe = build_temporal_features(
    #     [df_T1, df_T2, df_T3],
    #     ["T1", "T2", "T3"]
    # )
    #
    # target_dataframe = pd.read_excel(
    #     "quality_targets.xlsx",
    #     index_col=0
    # )
    #
    # common_index = X_dataframe.index.intersection(
    #     target_dataframe.index
    # )
    #
    # X = X_dataframe.loc[common_index].values
    #
    # y = target_dataframe.loc[
    #     common_index,
    #     ["N", "CP", "ADF", "NDF"]
    # ].values
    #
    # ------------------------------------------------------------------------

    # Synthetic data are used only to verify that the script runs.
    n_samples = 200
    n_time = 3
    feat_per_time = 9
    n_features = n_time * feat_per_time

    rng = np.random.default_rng(42)

    X_demo = rng.normal(
        size=(n_samples, n_features)
    )

    y_demo = np.zeros(
        (n_samples, 4),
        dtype=np.float64
    )

    # Synthetic relationships for demonstration only.
    latent_bio = (
        0.60 * X_demo[:, 0]
        + 0.25 * X_demo[:, 9]
        + 0.15 * X_demo[:, 18]
    )

    latent_struc = (
        0.15 * X_demo[:, 3]
        + 0.25 * X_demo[:, 12]
        + 0.60 * X_demo[:, 21]
    )

    y_demo[:, 0] = (
        3.8
        + 0.35 * latent_bio
        - 0.15 * latent_struc
        + rng.normal(0, 0.10, n_samples)
    )

    y_demo[:, 1] = (
        6.25 * y_demo[:, 0]
        + rng.normal(0, 0.35, n_samples)
    )

    y_demo[:, 2] = (
        25.0
        - 1.10 * latent_bio
        + 1.80 * latent_struc
        + rng.normal(0, 0.80, n_samples)
    )

    y_demo[:, 3] = (
        39.0
        - 1.70 * latent_bio
        + 2.60 * latent_struc
        + rng.normal(0, 1.10, n_samples)
    )

    # ------------------------------------------------------------------------
    # Independent train/test split
    # ------------------------------------------------------------------------

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_demo,
        y_demo,
        test_size=0.25,
        random_state=42,
        shuffle=True
    )

    # ------------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------------

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(
        X_train_raw
    )

    X_test = scaler_X.transform(
        X_test_raw
    )

    y_train = scaler_y.fit_transform(
        y_train_raw
    )

    # ------------------------------------------------------------------------
    # Train PI-SSN
    # ------------------------------------------------------------------------

    model, pred_train_scaled, pred_test_scaled, history = train_pi_ssn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=None,
        n_time=3,
        feat_per_time=9,
        epochs=600,
        lr=0.008,
        batch_size=32,
        validation_fraction=0.20,
        lambda_phy=0.5,
        lambda_l1=0.01,
        gamma_struc=1.0,
        scheduler_patience=20,
        early_stopping_patience=60,
        random_state=42,
        verbose=True
    )

    # Return predictions to the original target scale.
    pred_train = scaler_y.inverse_transform(
        pred_train_scaled
    )

    pred_test = scaler_y.inverse_transform(
        pred_test_scaled
    )

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------

    print("\nTraining-set performance:")
    training_metrics = calculate_regression_metrics(
        y_train_raw,
        pred_train
    )
    print_regression_metrics(training_metrics)

    print("\nIndependent test-set performance:")
    test_metrics = calculate_regression_metrics(
        y_test_raw,
        pred_test
    )
    print_regression_metrics(test_metrics)

    # ------------------------------------------------------------------------
    # Attention interpretation
    # ------------------------------------------------------------------------

    bio_attention, struc_attention = extract_attention_weights(
        model,
        X_test
    )

    mean_bio_attention = bio_attention.mean(axis=0)
    mean_struc_attention = struc_attention.mean(axis=0)

    print("\nMean temporal attention weights:")
    print("Time points:          T1       T2       T3")
    print(
        "Biochemical branch:",
        " ".join(
            f"{value:8.3f}"
            for value in mean_bio_attention
        )
    )
    print(
        "Structural branch: ",
        " ".join(
            f"{value:8.3f}"
            for value in mean_struc_attention
        )
    )

    # ------------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------------

    checkpoint_path = "PI_SSN_best.pth"

    save_pi_ssn_checkpoint(
        model=model,
        file_path=checkpoint_path,
        extra_information={
            "target_order": TARGET_NAMES,
            "time_points": ["T1", "T2", "T3"],
            "feature_names": FEATURE_NAMES,
            "best_validation_loss": min(
                history["validation_loss"]
            )
        }
    )

    print(
        f"\nModel checkpoint saved to: {checkpoint_path}"
    )

    # ------------------------------------------------------------------------
    # Load model and perform inference
    # ------------------------------------------------------------------------

    inference_model, checkpoint = load_pi_ssn_checkpoint(
        checkpoint_path,
        device="cpu"
    )

    X_new = X_test[:5]
    X_new_tensor = torch.from_numpy(
        X_new.astype(np.float32)
    )

    with torch.no_grad():
        pred_new_scaled = (
            inference_model(X_new_tensor)
            .cpu()
            .numpy()
        )

    pred_new = scaler_y.inverse_transform(
        pred_new_scaled
    )

    print("\nPredictions for five new samples:")
    print("Column order: N, CP, ADF, NDF")
    print(pred_new)
