find . -name "__pycache__" -type d -exec rm -r {} +
find . -name "*.pyc" -delete

rm -rf dist/ build/ SFOF4S.egg-info/
rm -rf dist/
python3 -m build

python3 -m twine upload dist/*
