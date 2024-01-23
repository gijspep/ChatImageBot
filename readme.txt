a chat bot that makes images, works with AWQ LLM models and SD/SDXL diffusion models(single ".safetensor" files)

setup:

1.) clone the repo.

2.) setup a venv.
    python3 -m venv venv
    where the second venv is the name of/path to your venv.
    https://docs.python.org/3/library/venv.html  

3.) pip install -r "requirements.txt"

4.) run chatImageBot.py it will create a folder structure,
    then place models/prompts in their respective dirs.

5.) activate your created venv and run the ChatImageBot.py file.
    source venv/bin/activate where venv is the name of your venv.
