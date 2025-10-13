from lib.pre_process import load_and_preprocess_data
from lib.models import NeuralNetwork, NLL
from lib.train import train_model_with_config
from lib.grid_search import GridSearch, SearchSpace
from lib.utils import decorate_optimizer
from torch.utils.data import DataLoader
import torch
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

path = "dataset/Breast Cancer METABRIC.csv"

processed_data = load_and_preprocess_data(path)
df = processed_data["df"]
numeric_cols = processed_data["numeric_cols"]
categorical_cols = processed_data["categorical_cols"]
sets = processed_data["sets"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    sets["train"],
    batch_size=64,
    shuffle=True,
)
val_loader = DataLoader(
    sets["val"],
    batch_size=64,
    shuffle=False,
)
test_loader = DataLoader(
    sets["test"],
    batch_size=len(sets["test"]),
    shuffle=False,
)


search_space = SearchSpace.model_validate(
    {
        "learning_rates": [5e-5, 1e-3, 5e-3, 1e-2, 5e-2],
        "weight_decays": [1e-4, 1e-5, 1e-6, 1e-7],
        "optimizers": [
            decorate_optimizer(torch.optim.Adam),
            decorate_optimizer(torch.optim.SGD),
        ],
        "num_epochs": [50, 100, 150],
    }
)

grid_searcher = GridSearch(search_space, device=device)
grid_searcher(
    Model=NeuralNetwork,
    model_init_args={
        "df": df,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "bias": True,
    },
    train_fn=train_model_with_config,
    loss_fn=NLL(),
    train_loader=train_loader,
    validation_loader=val_loader,
    test_loader=test_loader,
)
