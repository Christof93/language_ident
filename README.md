# Language identification with Python
Reimplementation of automatic language identification for demonstration purposes with minimal usage of existing libraries. (numpy)

### Testing the code
The code consists of the Naive Bayes model class (nb_classifier.py), the feature extraction utilities (features.py) and the anlysis notebook (test_li.ipynb).
In the notebook the usage of the code is walked through step-by-step with some explanations.
Also, the model is trained on a dataset imported from the HuggingFace repository and evaluated against the langid and langdetect libraries.

To try out the code follow these steps:

1. set up a python 3 virtual environment 

```
python3 -m venv language_ident_env
```

2. install the required packages to view the notebook 

```
python3 -m pip install -r requirements_analysis.txt
```
or if you just want to test the Naive Bayes model without the analysis notebook, you only need two required packages.
```
python3 -m pip install -r requirements_method.txt
``` 

3. Run the notebook 
```
jupyter notebook test_li.ipynb
```
or test the code by importing nb_classifier.py and features.py
