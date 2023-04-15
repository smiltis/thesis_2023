# thesis_2023

<!-- For installing the libraries -->

pip install -r requirements.txt --user

<!-- For requirements generation -->

pip show pipreqs <!-- if pipreqs already exist skip next line -->
pip install pipreqs --user
python.exe -m pipreqs.pipreqs . --force <!-- Generate the requirements.txt -->
