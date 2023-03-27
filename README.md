# Fiber Photometry GUI by CU Boulder's Donaldson Lab

## Installation Instructions

**New Users: Follow the instructions below to install the GUI**
> Our code utilizes Python and numerous packages within it, if you do not already have Python installed on your device, please refer to these links before continuing: <br>
https://wiki.python.org/moin/BeginnersGuide/Download \
https://docs.anaconda.com/anaconda/install/ \
https://pip.pypa.io/en/stable/installation/

*Note: this application was developed and tested with Python version 3.9, so it is recommended to install this version or higher in your environment.*

#Download Code
1. Navigate to https://github.com/donaldsonlab/PhAT
2. Click on the green button labeled “Code” located at the top right corner of the repository, then click on “Download ZIP” (Ensure that this file is saved locally on your device i.e., not on any cloud environments).
3. Locate the downloaded zip file on your device and place it somewhere convenient to easily navigate to it again. Avoid cloud storage. 
4. Unzip the file by right clicking on it and selecting unzip or use an unzipping utility (e.g., WinRAR on Windows systems). 
5. Take note of the FiberPho_Main folder location (folder path needed later).
    a. Mac/Unix: Right click on the folder, Hold the Option key, and copy “PhAT” as Pathname.
    b. Windows: Right click on the folder, select Properties, and take note of the text written next to Location on your computer, this is the folder’s path.

#Create Virtual Environment
Using Anaconda (Option 1: Recommended)
6a. Open a new terminal window (Mac/Unix) or Anaconda Prompt (not Anaconda 	Navigator) (Windows).
7a. Navigate to the location of the “PhAT” folder (noted from Step 3).
i.	Type the following command, instead typing your folder path within the brackets: “cd [path_to_PhAT_folder]”. Then hit enter.
    Ex. cd Desktop/DonaldsonLab/PhAT 
8a. Create a virtual environment and give it a name (e.g. “my_gui_env”) with the following command:
“conda create -n [your_env_name] python=[version] pip”. Then hit 	enter.
    Ex: conda create -n my_gui_env python=3.9 pip
9a. Activate the virtual environment with the following command:.
“conda activate [your_env_name]” Then hit enter.
    Ex: conda activate my_gui_env
10a. Execute the following commands to install dependencies.
    i.Type “pip list”. Then hit enter.
    No dependencies should be present since this is a new environment. 
    ii.	Type “pip install -r requirements.txt”. Then hit enter.
    iii.Type “pip list”. Then hit enter
    All necessary dependencies should now be installed.

#Using PIP/PyPI (Option 2)
6b. Open a new terminal window (command prompt for Windows)
7b. Navigate to the location of the “PhAT” folder (noted from Step 1C3).
    i. Type the following command, instead typing your folder path within the brackets: “cd [path_to_PhAT_folder]”. Then hit enter.
    Ex: cd Desktop/DonaldsonLab/PhAT
8b. Create a virtual environment and give it a name (e.g. “my_gui_env”) using one of the following commands.
    i. Mac/Unix: “python3 -m venv [your_env_name]”. Then hit enter.
    ii. Windows: “py -m venv [your_env_name]”. Then hit enter.
9b. Activate the virtual environment.
    i. Mac/Unix: “source [your_env_name]/bin/activate”. Then hit enter.
    ii. Windows: “.\[your_env_name]\Scripts\activate”. Then hit enter.
10b. Execute the following commands to install dependencies.
    i. Type “pip list”. Then hit enter.
    No dependencies should be present since this is a new environment. 
    ii. Type “pip install -r requirements.txt”. Then hit enter.
    iii. Type “pip list”. Then hit enter.
    All necessary dependencies should now be installed.
