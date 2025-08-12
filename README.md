Before use:
deschide-ti un cmd ca admin
cd in folder
setup_env.bat

deschideti cu vs code
rulati, daca device e cpu

in terminal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.version.cuda)"

cuda ar trebui sa fie v12.1 daca totul e ok cand il rulati iar ar trebui sa apara logurile din figura
<img width="734" height="234" alt="image" src="https://github.com/user-attachments/assets/c319d420-7e17-4dc7-9c86-c15d2b2a8f10" />

respectiv interfata grafica.
Clientul trebuie sa fie in: 1920x1080 , scaling 135%
Incercati ca regiunea de detectie sa fie pozitionata deasupra apei, ca in exemplu:

<img width="940" height="358" alt="image" src="https://github.com/user-attachments/assets/7ffaedc1-f896-4c68-a5db-bab3f6832bf7" />

