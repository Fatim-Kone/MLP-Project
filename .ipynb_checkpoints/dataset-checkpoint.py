{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cebb73ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "class PendulumDataset(Dataset):\n",
    "    def __init__(self, x, y):\n",
    "        self.x = torch.tensor(x, dtype=torch.float32)\n",
    "        self.y = torch.tensor(y, dtype=torch.float32)\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.x)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        return self.x[idx], self.y[idx]"
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
