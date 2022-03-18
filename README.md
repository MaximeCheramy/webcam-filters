# Webcam Filters

## Installation

Create a virtual camera using [v4l2loopback](https://github.com/umlaeute/v4l2loopback):

```
modprobe v4l2loopback exclusive_caps=1 video_nr=8 card_label="Virtual Camera" 
```

Dependencies are managed using [Poetry](https://python-poetry.org). Install dependencies using:

```
poetry install
```

## Run

poetry run python main.py
