import pandas as pd
from be_great import GReaT

df = pd.read_csv("your_file.csv")

model = GReaT(llm="distilgpt2", batch_size=32, epochs=25)
model.fit(df)
synthetic = model.sample(n_samples=100)
