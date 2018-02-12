# dl-dataday-workshop

Please make sure you do the setup below before coming to class:
First (optional), create your EC2 account on AWS, if you don't have it.
Second, install Anaconda with Python 3.5 
Third, download and run the following
 
git clone https://github.com/humayun/dl-dataday-workshop.git
cd dl-dataday-workshop

For EC2 (Linux/Ubuntu) User:

conda env create -f dl_workshop_ec2.yml

For Mac User:

conda env create -f dl_workshop_mac.yml

For Window User: 

conda env create -f dl_workshop_windows.yml
 
If you don't want to install anaconda then, you need to install following library:
 
pip install pandas

pip install pillow

pip install opencv-python

pip install scikit-image

pip install scikit-learn

pip install h5py

pip install tensorflow

pip install keras

pip install seaborn
 
fourth, read instruction in project folder inside code and download data prior to workshop.

 
If you run in to trouble:

1) Check your version of python! Consider reinstalling python.

2) Make a clean amazon instance on EC2. You can purchase p2.xlarge machines with a fast GPU for under $8 for the day.

3) If all else fails, please come early
