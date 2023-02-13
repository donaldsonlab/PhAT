# Fiber Photometry GUI by CU Boulder's Donaldson Lab

## Installation Instructions

**Current Users: Repeat Step 1 and activate as usual when a new update of the FiberPho GUI has been released** \
**New Users: Follow the instructions below to install the GUI**
> Our code utilizes Python and numerous packages within it, if you do not already have Python installed on your device, please refer to these links before continuing: <br>
https://wiki.python.org/moin/BeginnersGuide/Download \
https://docs.anaconda.com/anaconda/install/ \
https://pip.pypa.io/en/stable/installation/

*Note: this application was developed and tested with Python version 3.9, so it is recommended to install this version or higher in your environment.*

1. Download Code
- Click on the green button labeled "Code" located at the top right corner of this repository and click on Download ZIP (Ensure that this file is saved locally on your device, in other words not on any cloud environments).
- Go to the downloaded zip file on your device and place it somewhere convenient to you so that you may easily navigate to it, again avoiding cloud storage.
- Unzip the file by clicking on it or using an unzipping utility (e.g. WinRAR), and take note of the FiberPho_Main folder's path, we will need this later.)
    - Mac/Unix: Right click folder &rarr; Hold Option key &rarr; Click Copy "FiberPho_Main" as Pathname.
    - Windows: Right click folder &rarr; Properties &rarr; Take note of text next to "Location:"

2. Create Virtual Environment
    > We will be creating a virtual environment to ensure that there are no issues with any currently installed python packages/modules. You only need to complete these steps once. You may install the GUI using either Anaconda or PIP/PyPI. We recommend using Anaconda to utilize Jupyter Notebook for ease of use and inline error logging. The instructions for both are below:
    - **Using PIP/PyPI:**
        1. Open a new terminal window and navigate to the location of the fiberpho_gui folder (from step 1C)
            - Type "cd path_to_fiberpho_gui_folder" and ensure you are in the write directory
            - Ex: "cd Desktop/DonaldsonLab/fiberpho_gui"
        2. Create a virtual environment with a name (e.g. "gui_env") using one of the following commands: 
            - Mac/Unix: `python3 -m venv gui_env`
            - Windows: `py -m venv gui_env`
        3. Activate the virtual environment:
            - Mac/Unix: `source gui_env/bin/activate`
            - Windows: `.\gui_env\Scripts\activate`
        4. Confirm you are in the virtual environment/directory:
            - Mac/Unix: `which python`
                - Ex. directory: `.../env/bin/python`
            - Windows: `where python` 
                - Ex. directory: `...\env\Scripts\python.exe`
            - *You should be in the given 'environment_name' directory* 
        5. Execute the following command: `pip3 install -r requirements.txt`
            > Type "pip list" to ensure all necessary dependencies are installed
    - **Using Anaconda:**
        1. Open a new terminal window(Mac/Unix) or Anaconda prompt(Windows)
        2. Navigate to the location of your fiberpho_gui folder (from step 1C).
            - Type "cd path_to_fiberpho_gui_folder" and ensure you are in the right directory
            - Ex - "cd Desktop/DonaldsonLab/fiberpho_gui"
        3. Create a virtual environment with the following command:
            - `conda create -n [Environment Name] python=<version> pip`
            - Ex: `conda create -n gui_env python=3.9 pip`
        4. Activate the virtual environment:
            - `conda activate [Environment Name]`
        5. Execute the following command: `pip install -r requirements.txt`
            > Type "pip list" to ensure all necessary dependencies are installed

3. Activate Virtual Environment
    > You must always activate the environment you created with its given name in order to run the GUI.
- PIP Users Example: 
    - Mac/Unix: `source gui_env/bin/activate`
    - Windows: `.\gui_env\Scripts\activate`
- Anaconda Users Example:
    - `conda activate gui_env`

**The FiberPho GUI should now be installed and ready to use**

You have the option of utilizing the GUI as a script (*recommended*) through your terminal/command prompt or using Jupyter Notebook. The instructions for each are below -

**Run with Python Script:** \
If you would like to deploy the server through your terminal or command prompt, follow the below instructions:
In your terminal/prompt, navigate to the location of the `FiberPho_Main` folder and run the following command -

`panel serve --show FiberGuiScript.py --websocket-max-message-size=104876000 --autoreload`

This command will launch the GUI in a new browser window or tab. \
To properly shutdown the GUI, press `Ctrl + C` on your keyboard. 
> Note: Any code changes made to the `.py` file will refresh the entire server instance. To avoid this, omit the `--autoreload` argument.

**Run with Jupyter Notebook:** \
If you would like to utilize Jupyter Notebook to deploy the server, simply navigate to the `FiberPho_Main` folder then simply run the `jupyter lab` command in your commandline or however else you run Jupyter. Open the notebook (.ipynb) file and begin to execute each cell from the top, *making sure* to let each cell finish execution before continuing to the next. Upon execution of the second to last cell, a local URL will print below the cell that navigates to the interface `(e.g. http://localhost:#####)`. \
To properly shutdown the GUI, run the last cell in the notebook containing the line `server.stop()`. 
> If using Anaconda, any errors that may arise will be displayed under the current cell in your notebook. 
