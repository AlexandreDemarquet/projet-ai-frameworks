{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "PermissionError",
     "evalue": "[Errno 13] Permission denied: '/app'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mPermissionError\u001b[0m                           Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[3], line 8\u001b[0m\n\u001b[1;32m      5\u001b[0m index_path \u001b[38;5;241m=\u001b[39m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(index_dir, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmy_index.ann\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# S'assurer que le dossier existe\u001b[39;00m\n\u001b[0;32m----> 8\u001b[0m \u001b[43mos\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmakedirs\u001b[49m\u001b[43m(\u001b[49m\u001b[43mindex_dir\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mexist_ok\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m)\u001b[49m\n\u001b[1;32m     10\u001b[0m \u001b[38;5;66;03m# Création de l'index Annoy\u001b[39;00m\n\u001b[1;32m     11\u001b[0m dimension \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m128\u001b[39m\n",
      "File \u001b[0;32m~/miniforge3/envs/optim/lib/python3.10/os.py:215\u001b[0m, in \u001b[0;36mmakedirs\u001b[0;34m(name, mode, exist_ok)\u001b[0m\n\u001b[1;32m    213\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m head \u001b[38;5;129;01mand\u001b[39;00m tail \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m path\u001b[38;5;241m.\u001b[39mexists(head):\n\u001b[1;32m    214\u001b[0m     \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 215\u001b[0m         \u001b[43mmakedirs\u001b[49m\u001b[43m(\u001b[49m\u001b[43mhead\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mexist_ok\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mexist_ok\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    216\u001b[0m     \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mFileExistsError\u001b[39;00m:\n\u001b[1;32m    217\u001b[0m         \u001b[38;5;66;03m# Defeats race condition when another thread created the path\u001b[39;00m\n\u001b[1;32m    218\u001b[0m         \u001b[38;5;28;01mpass\u001b[39;00m\n",
      "File \u001b[0;32m~/miniforge3/envs/optim/lib/python3.10/os.py:225\u001b[0m, in \u001b[0;36mmakedirs\u001b[0;34m(name, mode, exist_ok)\u001b[0m\n\u001b[1;32m    223\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m\n\u001b[1;32m    224\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 225\u001b[0m     \u001b[43mmkdir\u001b[49m\u001b[43m(\u001b[49m\u001b[43mname\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmode\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    226\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mOSError\u001b[39;00m:\n\u001b[1;32m    227\u001b[0m     \u001b[38;5;66;03m# Cannot rely on checking for EEXIST, since the operating system\u001b[39;00m\n\u001b[1;32m    228\u001b[0m     \u001b[38;5;66;03m# could give priority to other errors like EACCES or EROFS\u001b[39;00m\n\u001b[1;32m    229\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m exist_ok \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m path\u001b[38;5;241m.\u001b[39misdir(name):\n",
      "\u001b[0;31mPermissionError\u001b[0m: [Errno 13] Permission denied: '/app'"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# Chemin où l'index sera stocké\n",
    "index_dir = \"/app/annoy_index\"\n",
    "index_path = os.path.join(index_dir, \"my_index.ann\")\n",
    "\n",
    "# S'assurer que le dossier existe\n",
    "os.makedirs(index_dir, exist_ok=True)\n",
    "\n",
    "# Création de l'index Annoy\n",
    "dimension = 128\n",
    "index = AnnoyIndex(dimension, 'angular')\n",
    "\n",
    "# Vérifier si l'index existe déjà\n",
    "if os.path.exists(index_path):\n",
    "    index.load(index_path)\n",
    "    print(\"Index chargé avec succès !\")\n",
    "else:\n",
    "    print(\"Création d'un nouvel index...\")\n",
    "    for i in range(1000):\n",
    "        v = [i * 0.1 for _ in range(dimension)]\n",
    "        index.add_item(i, v)\n",
    "    index.build(10)\n",
    "\n",
    "    # Sauvegarde en s'assurant que le dossier existe\n",
    "    index.save(index_path)\n",
    "    print(\"Index sauvegardé avec succès.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "optim",
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
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
