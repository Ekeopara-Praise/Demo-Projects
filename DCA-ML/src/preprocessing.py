import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DCAPreprocessor:
    """Preprocessing pipeline for Neural Network-based Decline Curve Analysis (DCA).

    Filters out zero-rate / shut-in periods to train models strictly on active
    producing history.
    """

    def __init__(
        self,
        date_col="Date",
        rate_col="Oil_Rate_BOPD",
        pressure_col="THP_psi",
        seq_length=12,
        pred_length=3,
    ):
        self.date_col = date_col
        self.rate_col = rate_col
        self.pressure_col = pressure_col
        self.seq_length = seq_length
        self.pred_length = pred_length

        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def filter_zero_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows where oil rate is zero or missing, preserving active flow days only."""
        data = df.copy()

        # Ensure datetime format & chronological order
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        data = data.sort_values(by=self.date_col).reset_index(drop=True)

        # Drop zero or negative production rows
        data = data[data[self.rate_col] > 0].reset_index(drop=True)

        # Handle missing pressure/secondary metrics via linear interpolation along active days
        if self.pressure_col in data.columns:
            data[self.pressure_col] = data[self.pressure_col].interpolate(
                method="linear"
            )

        return data

    def remove_outliers(
        self, df: pd.DataFrame, window: int = 5, threshold: float = 3.0
    ) -> pd.DataFrame:
        """Filters short-term operational spikes using a rolling Z-score across producing days."""
        data = df.copy()
        rolling_mean = (
            data[self.rate_col].rolling(window=window, min_periods=1).mean()
        )
        rolling_std = (
            data[self.rate_col].rolling(window=window, min_periods=1).std()
        )

        rolling_std = rolling_std.replace(0, 1.0)
        z_scores = np.abs((data[self.rate_col] - rolling_mean) / rolling_std)

        # Replace extreme spikes with local rolling mean
        data[self.rate_col] = np.where(
            z_scores > threshold, rolling_mean, data[self.rate_col]
        )

        return data

    def add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers domain features: Cumulative Oil (Np) and Normalized Cumulative Producing Time."""
        data = df.copy()

        # Cumulative Oil Production (Integral signal)
        data["Cum_Oil_STB"] = data[self.rate_col].cumsum()

        # Active Producing Time Index (Sequential producing steps)
        data["Producing_Step"] = np.arange(1, len(data) + 1)
        data["Normalized_Time"] = data["Producing_Step"] / len(data)

        return data

    def scale_features(
        self, df: pd.DataFrame, feature_cols: list, is_training: bool = True
    ):
        """Scales inputs (X) and targets (Y) independently to [0, 1]."""
        data = df.copy()

        if is_training:
            scaled_x = self.scaler_x.fit_transform(data[feature_cols])
            scaled_y = self.scaler_y.fit_transform(data[[self.rate_col]])
        else:
            scaled_x = self.scaler_x.transform(data[feature_cols])
            scaled_y = self.scaler_y.transform(data[[self.rate_col]])

        return scaled_x, scaled_y

    def create_sequences(self, X_data: np.ndarray, Y_data: np.ndarray):
        """Generates 3D tensors (N_samples, seq_length, num_features) using a sliding window."""
        X_seq, Y_seq = [], []
        window_size = self.seq_length + self.pred_length

        for i in range(len(X_data) - window_size + 1):
            x_window = X_data[i : i + self.seq_length, :]
            y_window = Y_data[
                i + self.seq_length : i + self.seq_length + self.pred_length, 0
            ]

            X_seq.append(x_window)
            Y_seq.append(y_window)

        return np.array(X_seq), np.array(Y_seq)

    def inverse_transform_rates(self, scaled_rates: np.ndarray) -> np.ndarray:
        """Converts normalized predictions back to BOPD."""
        if scaled_rates.ndim == 1:
            scaled_rates = scaled_rates.reshape(-1, 1)
        return self.scaler_y.inverse_transform(scaled_rates)

    def fit_transform_pipeline(
        self, df: pd.DataFrame, feature_cols: list
    ) -> tuple:
        """Executes full pipeline: Zero filtering -> Outlier cleaning -> Engineering -> Scaling -> Windowing."""
        df_active = self.filter_zero_production(df)
        df_filtered = self.remove_outliers(df_active)
        df_engineered = self.add_physics_features(df_filtered)

        scaled_x, scaled_y = self.scale_features(
            df_engineered, feature_cols, is_training=True
        )
        X_seq, Y_seq = self.create_sequences(scaled_x, scaled_y)

        return X_seq, Y_seq, df_engineered