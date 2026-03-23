# InterviewHelper
**AY-AllenYe @ HDU**

Due to the former competition which I chose paraformer (FunASR) to participated, I decided to create a app to help Automatic Speech Recognition (known as ASR).

Wish this project helps you. And your Star is really helpful.

## version 1.x

### Explanation

**1.Baseline**

**2.Pre-trained Models**




```
the file tree
|-- (root)
    |-- baseline.py
    |-- main.py
    |-- models
    |   |-- (Pre-trained or finetuned models stored here)
    |-- utils
    |   |-- logger.py                   // Record and Report (Auto).
    |   |-- mic_test.py                 // Verify whether the microphone of user's computer is functional.
```

### Bash

​	It is strongly advised to verify the availability of microphone.

```
cd (root)
python utils/mic_test.py
```

​	Then you can run the main Python file.

```
python main.py
```

​	The users are probably change the path.

### Dependency

​	In my machine these wheels is tested related.

```
funasr == 1.3.1
numpy == 1.24.0
sounddevice == 0.5.5
scipy == 1.15.3
and others.
```

### Updated Logs

#### 2025.12.29