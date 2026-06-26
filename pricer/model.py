import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer
from pathlib import Path


class ResidualBlock(nn.Module):
    """A residual block with two linear layers, LayerNorm, ReLU, and a skip connection."""

    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class DeepNeuralNetwork(nn.Module):
    """
    Deep price-regression network with residual connections.

    Architecture:
        input_layer → N residual blocks → output_layer (scalar)
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 10,
        hidden_size: int = 4096,
        dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2)]
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class DeepNeuralNetworkRunner:
    """
    Orchestrates training, evaluation, saving, loading, and inference
    for the DeepNeuralNetwork price predictor.

    Targets are log-normalised before training and exponentiated back
    during evaluation so that MAE is expressed in dollars.
    """

    def __init__(self, train: list, val: list):
        self.train_data = train
        self.val_data = val

        # Populated by setup()
        self.vectorizer: HashingVectorizer | None = None
        self.model: DeepNeuralNetwork | None = None
        self.device: torch.device | None = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader: DataLoader | None = None
        self.X_val: torch.Tensor | None = None
        self.y_val: torch.Tensor | None = None
        self.y_val_norm: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def setup(self):
        """Vectorise text, build model, configure optimiser and data loaders."""
        # ---- Vectorise ----
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)

        train_docs = [item.summary for item in self.train_data]
        X_train_np = self.vectorizer.fit_transform(train_docs)
        X_train = torch.FloatTensor(X_train_np.toarray())
        y_train_raw = torch.FloatTensor([float(item.price) for item in self.train_data]).unsqueeze(1)

        val_docs = [item.summary for item in self.val_data]
        X_val_np = self.vectorizer.transform(val_docs)
        self.X_val = torch.FloatTensor(X_val_np.toarray())
        self.y_val = torch.FloatTensor([float(item.price) for item in self.val_data]).unsqueeze(1)

        # ---- Log-normalise targets ----
        y_train_log = torch.log(y_train_raw + 1)
        y_val_log = torch.log(self.y_val + 1)
        self.y_mean = y_train_log.mean()
        self.y_std = y_train_log.std()
        y_train_norm = (y_train_log - self.y_mean) / self.y_std
        self.y_val_norm = (y_val_log - self.y_mean) / self.y_std

        # ---- Model ----
        self.model = DeepNeuralNetwork(X_train.shape[1])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"DeepNeuralNetwork created — {total_params:,} trainable parameters")

        self.device = self._select_device()
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # ---- Training plumbing ----
        self.loss_function = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

        train_dataset = TensorDataset(X_train, y_train_norm)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, epochs: int = 5):
        """Run the training loop for the given number of epochs."""
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(self.X_val.to(self.device))
                val_loss = self.loss_function(val_preds, self.y_val_norm.to(self.device))
                val_preds_orig = torch.exp(val_preds * self.y_std + self.y_mean) - 1
                mae = torch.abs(val_preds_orig - self.y_val.to(self.device)).mean()

            print(
                f"Epoch [{epoch}/{epochs}]  "
                f"Train Loss: {np.mean(train_losses):.4f}  "
                f"Val Loss: {val_loss.item():.4f}  "
                f"Val MAE: ${mae.item():.2f}  "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )
            self.scheduler.step()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path):
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str | Path):
        """Load model weights from disk onto the current device."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def inference(self, item) -> float:
        """
        Predict the price of a single Item.

        Args:
            item: An Item object with a populated .summary field.

        Returns:
            Predicted price in dollars (≥ 0).
        """
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([item.summary])
            tensor = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(tensor)[0]
            price = torch.exp(pred * self.y_std + self.y_mean) - 1
        return max(0.0, price.item())
