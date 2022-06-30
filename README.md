# cmpt353

# To run code, will require all library installs:
pip3 install selenium
pip3 install pandas
pip3 install numpy
pip3 install matplotlib
pip3 install seaborn
pip3 install -U scikit-learn
pip3 install statsmodels
pip3 install wordcloud
pip3 install --user -U nltk
pip3 install xgboost

# May require additional install to use xgboost
brew install libomp

# DATA SCRAPING
# For running the scraper file, need to install the chrome driver
# After installing the chrome driver, the path of chrome driver needs to be used on the python file
# Run the python script to output (Name of the raw file).csv file which will used in data cleaning process.
# NOTE: scraped_jobs.csv data file already available in folder

# DATA CLEANING
# Command to output Cleaned_Data_final.csv:
# NOTE: Cleaned_Data_final.csv is already available in folder
python3 Data_Cleaning.py scraped_jobs.csv

# EXPLORATORY DATA ANALYSIS
# Uses the Cleaned_Data_final.csv file to produce plots
# Exploratory_data_analysis.ipynb to be run in Jupyter notebook

# MODEL BUILDING
# Command to run main analysis:
python3 Model_Building.py Cleaned_Data_final.csv

# NOTE TO EVALUATORS:
# All commits were made from one account (kkuninaka) due to technical issues