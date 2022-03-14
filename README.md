# Webcam Filters

## Installation

Create a virtual camera using [v4l2loopback](https://github.com/umlaeute/v4l2loopback):

```
modprobe v4l2loopback exclusive_caps=1 video_nr=8 card_label="Virtual Camera" 
```

Install dependencies:

```
pip3 install -r requirements.txt
```

## Run

python3 main.py
