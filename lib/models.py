import pandas as pd
import torch
from torch import nn


class BaseSurvivalClass(nn.Module):
    def __init__(
        self, df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]
    ):
        super(BaseSurvivalClass, self).__init__()
        self.categorical_cols = categorical_cols
        self.numerical_cols = numeric_cols
        self.features = self.numerical_cols + self.categorical_cols
        self.embeddings = nn.ModuleDict()
        for col in self.categorical_cols:
            num_unique_values = int(df[col].nunique())
            embedding_size = 4
            self.embeddings[col] = nn.Embedding(num_unique_values, embedding_size)

    def embed(self, x):
        embedded_cols = []
        for col in self.categorical_cols:
            # ndarray = np.array()
            embedded_col = self.embeddings[col](x[:, self.features.index(col)].long())
            # print(embedded_col.shape)
            embedded_cols.append(embedded_col)
        numerical_data = torch.stack(
            [x[:, self.features.index(col)] for col in self.numerical_cols],
            dim=1,
        ).float()
        x = torch.cat(embedded_cols + [numerical_data], dim=1)
        return x

    def embed_with_time(self, x, t):
        time_data = torch.reshape(t, (x.shape[0], 1)).float()
        x = self.embed(x)
        x = torch.cat((x, time_data), dim=1)
        return x

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")


class NeuralNetwork(BaseSurvivalClass):
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
        numeric_cols: list[str],
        bias=True,
    ):
        super().__init__(df, categorical_cols, numeric_cols)

        self.net = nn.Sequential(
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(1, bias=bias),
        )

        # self.bias = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.LazyLinear(1))

    def forward(self, x):
        x = self.embed(x)
        return self.net(x).squeeze()


class NLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, failure_times, is_observed):
        # Number of observed events
        return torch.sum(torch.exp(preds) * failure_times - is_observed * preds, dim=0)


class ProfiledNLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, failure_times, is_observed):
        # Number of observed events
        d = torch.sum(is_observed)

        log_of_sum = torch.log(
            torch.sum(torch.exp(preds) * failure_times, dim=0) + 1e-8
        )
        sum_of_log = torch.sum(is_observed * preds, dim=0)

        nll = d * log_of_sum - sum_of_log
        return nll


class CoxPH(BaseSurvivalClass):
    def __init__(
        self, df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]
    ):
        super().__init__(df, categorical_cols, numeric_cols)
        self.net = nn.Sequential(
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        x = self.embed(x)
        return self.net(x).squeeze()


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards partial likelihood loss.
    Efficient implementation using the Breslow approximation for tied events.
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_h, durations, events):
        """
        Args:
            log_h: log hazard ratio predictions from model, shape (batch_size,)
            durations: observed time (either event or censoring time), shape (batch_size,)
            events: event indicator (1 if event, 0 if censored), shape (batch_size,)

        Returns:
            loss: negative log partial likelihood
        """
        # Sort by time
        sorted_durations, sorted_indices = torch.sort(durations)
        sorted_log_h = log_h[sorted_indices]
        sorted_events = events[sorted_indices]

        # Calculate negative partial log-likelihood
        log_likelihood = 0.0

        for i in range(len(sorted_durations)):
            if sorted_events[i] == 1:  # If an event occurred
                # Risk set: all individuals still at risk at time t_i
                # (includes individual i and all individuals with longer times)
                risk_set_log_h = sorted_log_h[i:]

                # Log partial likelihood contribution
                # log h_i - log(sum of exp(log h_j) for j in risk set)
                log_likelihood += sorted_log_h[i] - torch.logsumexp(
                    risk_set_log_h, dim=0
                )

        return -log_likelihood  # Return negative for minimization


class TiedCoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        preds: torch.tensor,
        failure_times: torch.tensor,
        is_observed: torch.tensor,
    ):
        unique_times = torch.unique(failure_times[is_observed == 1])
        # Sort by time
        sorted_times, sorted_indices = torch.sort(failure_times)
        sorted_preds = preds[sorted_indices]

        log_likelihood = 0.0

        for t in unique_times:
            indices_at_t = (sorted_times == t).nonzero(as_tuple=True)[0]
            preds_at_t = sorted_preds[indices_at_t]
            m_j = len(preds_at_t)
            sum_preds = torch.sum(preds_at_t)
            sum_log = 0.0
            indices_at_risk = (sorted_times >= t).nonzero(as_tuple=True)[0]
            preds_at_risk = sorted_preds[indices_at_risk]
            for l in range(m_j):
                log = torch.log(
                    torch.sum(torch.exp(preds_at_risk))
                    - (l / m_j) * torch.sum(torch.exp(preds_at_t))
                )
                sum_log += log
            log_likelihood += sum_preds - sum_log
        return -log_likelihood  # Return negative for minimization
