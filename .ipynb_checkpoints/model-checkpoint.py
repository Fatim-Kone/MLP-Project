{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c40c56a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "\n",
    "class LSTMModel(nn.Module):\n",
    "    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):\n",
    "        super(LSTMModel, self).__init__()\n",
    "        self.hidden_dim = hidden_dim\n",
    "        self.layer_dim = layer_dim\n",
    "        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)\n",
    "        self.fc = nn.Linear(hidden_dim, output_dim)\n",
    "\n",
    "    def forward(self, x, h0=None, c0=None):\n",
    "        if h0 is None or c0 is None:\n",
    "            h0 = torch.zeros(self.layer_dim, x.size(\n",
    "                0), self.hidden_dim).to(x.device)\n",
    "            c0 = torch.zeros(self.layer_dim, x.size(\n",
    "                0), self.hidden_dim).to(x.device)\n",
    "\n",
    "        out, (hn, cn) = self.lstm(x, (h0, c0))\n",
    "        out = self.fc(out[:, -1, :]) \n",
    "        return out, hn, cn"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
