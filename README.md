# bostondeployment

### Software and Tools Requirement

1. [Github Account]
2. [VSCodeIDE]
3. Heroku Account
4. GIT CLI

Create a new environment
conda create -p venv python==3.7 -y
conda activate venv/

For VS Code
python -m venv venv

git config --global user.name

pip install -r requirements.txt
pip install streamlit
streamlit run streamlit_app.py

docker build -t house_price .
docker run -p 5000:5000 house_price