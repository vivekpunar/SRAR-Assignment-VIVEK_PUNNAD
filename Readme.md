# SRAR Assignment

In this repository you will find the code to run a Mask detection model built on YoloV5 pytorch framework. It is highly recommended that you need to have a system which can install pytorch package.

## Steps to run the object detection file 

### 1. Download the repository on your system and unzip the repository.

### 2. Open Command line and navigate to the folder path.

### 3. Create a Virtual environment

To create the virtual environment in the command line enter the below command. It will create a virtual env 'srar' in your folder.
```
virtualenv srar
```

### 4. Activate the environment

To Activate the environment run the following command in cmd.

```
srar\Scripts\activate
```

### 5. Install the necessary packages

The below command will install all the necessary packages
```
pip install -r requirements.txt
```
### 6. Run the detection model

In the command line execute the python file
```
python detect_model.py
```

## Information about the detect_model.py file

The detect_model.py file will gain access to your webcam and start detecting mask and no mask labels. It will also count the number of people wearing and not wearing masks and display it on the screen.

#### If you want to change the source from webcam to a video file.

Open the detect_model.py file in your text editor or IDE and make changes to the line 33
```
source = '0'
```

to 

```
source = 'YOUR VIDEO FILE PATH'
```

The source is initially set to value '0' which is for webcam.

#### To close the detection screen.

While executing the code if you want to close the screen or video. Press 'Q' on your keyboard it will automatically end the detection. 
