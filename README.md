# open-shot-vidio-editor-multilingual-videos-with-lipsync

## Frontend
we are using Openshot-qt https://github.com/OpenShot/openshot-qt   

         git clone https://github.com/OpenShot/libopenshot-audio.git
        git clone https://github.com/OpenShot/libopenshot.git
        use openshot-qt folder from this repo

## Follow this to set up 
    https://github.com/OpenShot/openshot-qt/wiki/Become-a-Developer

## Backend (tool)
    clone this repo and run the following commands
        conda env create --name tool  --file=tool.yml     
        conda activate tool
    
    ngrok 
        ./ngrok config add-authtoken <TOKEN>
        ./ngrok http 3000

    python app.py


## Run Frontend
    PYTHONPATH=libopenshot/build/bindings/python  python3 openshot-qt/src/launch.py  https://bccf-14-139-82-6.ngrok-free.app  --no-sandbox


## Features of the editor 
    check Video_editor.pdf 
    For more details refer paper.pdf(Intelligent Video Editing: Incorporating Modern Talking Face
Generation Algorithms in a Video Editor)








<!-- ![Watch the video](https://iiitaphyd-my.sharepoint.com/:v:/g/personal/aparna_agrawal_research_iiit_ac_in/EUuVCQWOPghPvXcSYVD9p54BuTU61o0etHCFFjr0prkg4g?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0RpcmVjdCJ9fQ&e=93acrO) -->

