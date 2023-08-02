# Fiber Photometry GUI by CU Boulder's Donaldson Lab

**A detailed protocol for the installation, use and modifcation of this software can be found in Current Protocols in Neuroscience at: https://doi.org/10.1002/cpz1.763**


## Installation Instructions

**New Users: Follow the instructions below to install the GUI**
> Our code utilizes Python and numerous packages within it, if you do not already have Python installed on your device, please refer to these links before continuing: <br>
https://wiki.python.org/moin/BeginnersGuide/Download \
https://docs.anaconda.com/anaconda/install/ \
https://pip.pypa.io/en/stable/installation/

*Note: this application was developed and tested with Python version 3.9, so it is recommended to install this version or higher in your environment.*

**Download Code**
1. Navigate to https://github.com/donaldsonlab/PhAT
2. Click on the green button labeled “Code” located at the top right corner of the repository, then click on “Download ZIP” (Ensure that this file is saved locally on your device i.e., not on any cloud environments).
3. Locate the downloaded zip file on your device and place it somewhere convenient to easily navigate to it again. Avoid cloud storage. 
4. Unzip the file by right clicking on it and selecting unzip or use an unzipping utility (e.g., WinRAR on Windows systems). 
5. Take note of the FiberPho_Main folder location (folder path needed later). \
&ensp; a. Mac/Unix: Right click on the folder, Hold the Option key, and copy “PhAT” as Pathname. \
&ensp; b. Windows: Right click on the folder, select Properties, and take note of the text written next to Location on your computer, this is the folder’s path. 

**Create Virtual Environment**
Using Anaconda (Option 1: Recommended)
1. Open a new terminal window (Mac/Unix) or Anaconda Prompt (not Anaconda 	Navigator) (Windows).
2. Navigate to the location of the “PhAT” folder (noted in Step 3) using the command: \
&ensp; “cd [path_to_PhAT_folder]” \
&ensp; Ex: cd Desktop/DonaldsonLab/PhAT 
3. Create a virtual environment: \
&ensp; “conda create -n [your_env_name] python=[version] pip” \
&ensp;    Ex: conda create -n my_gui_env python=3.9 pip 
4. Activate the virtual environment with the following command: \
&ensp;  “conda activate [your_env_name]” \
&ensp;   Ex: conda activate my_gui_env
5. Update conda  with the following command: \
&ensp; "conda update -n base -c defaults conda" 
7. Install jupyter lab or jupyter notebook using the following commands:  \
&ensp; "conda install -c conda-forge jupyterlab" and/or "conda install -c conda-forge notebook"
8. Execute the following command to install dependencies: \
&ensp; “pip install -r requirements.txt” 
9. Optional* Add the tmac package (https://github.com/Nondairy-Creamer/tmac) for motion correction: \
&ensp; "cd Fiberpho_main" \
&ensp; "git clone https://github.com/Nondairy-Creamer/tmac" \
&ensp; "cd tmac" \
&ensp; "pip install -e ." \
&ensp; cd ..&#92;..
9. Launch jupyter lab or notebook from the terminal by typing one of the following commands: \
&ensp; "jupyter lab" \
&ensp; "jupyter notebook" \
&ensp; Note: If you choose to launch these applications of Anaconda Navigator be sure to change the environment on the Anaconda Navigator from base(root) to your new environment. 


**Using PIP/PyPI (Option 2)**
1. Open a new terminal window (command prompt for Windows)
2. Navigate to the location of the “PhAT” folder (noted in Step 3) following command: \
&ensp;    “cd [path_to_PhAT_folder]” \
&ensp;  Ex: cd Desktop/DonaldsonLab/PhAT
3. Create a virtual environment and give it a name (e.g. “my_gui_env”) using one of the following commands. \
&ensp;    i. Mac/Unix: “python3 -m venv [your_env_name]” \
&ensp;    ii. Windows: “py -m venv [your_env_name]”
4. Activate the virtual environment. \
&ensp;    i. Mac/Unix: “source [your_env_name]/bin/activate” \
&ensp;    ii. Windows: “.\[your_env_name]\Scripts\activate” 
5. Execute the following commands to install dependencies. \
&ensp;     “pip install -r requirements.txt”
6. Launch the GUI by executing the following bash script in your terminal: \
&ensp;     "bash gui_script.sh"
8. Optional* Add the tmac package (https://github.com/Nondairy-Creamer/tmac) for motion correction with the following commands: \
&ensp; "cd Fiberpho_main" \
&ensp; "git clone https://github.com/Nondairy-Creamer/tmac" \
&ensp; "cd tmac" \
&ensp; "pip install -e ." \
&ensp; cd ..&#92;..
