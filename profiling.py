import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from src.pokemon_classification.data import PokemonDataset, pokemon_data
from src.pokemon_classification.model import MyAwesomeModel

# Register custom class for safe deserialization
torch.serialization.add_safe_globals({"PokemonDataset": PokemonDataset})

print("Profiling model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = MyAwesomeModel().to(DEVICE)
model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
model.eval()

train_set, test_set = pokemon_data()
test_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

# Profiling without using tensorboard-plugin-profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if DEVICE.type == "cuda" else [ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
