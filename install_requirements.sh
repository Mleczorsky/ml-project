# Create new requirements.txt
# pip freeze > requirements.txt

# Mozliwe ze trzeba:
#   pip install virtualenv
virtualenv .env && source .env/bin/activate && pip install -r requirements.txt
