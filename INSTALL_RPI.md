## Configurar Raspberry Pi 4

Equipamento testado: Raspberry Pi 4 com o [HAT ReSpeaker](https://wiki.seeedstudio.com/ReSpeaker_6-Mic_Circular_Array_kit_for_Raspberry_Pi/) (array circular de 6 microfones).


##### 1. Instalar o OS num cartão.
Usar o [Raspberry Pi Imager](https://www.raspberrypi.org/blog/raspberry-pi-imager-imaging-utility/) para instalar uma imagem do Raspberry Pi OS num cartão microSD.  
No instalador escolher RPI OS (32-BIT).
##### 2. Setup inicial.
Colocar o cartão num RPI e fazer setup inicial, definindo localização, password (no projeto usou-se user: pi e pass: pmsobral), etc.
##### 3. Depois de reboot, instalar updates adicionais
    sudo apt update  
    sudo apt upgrade

##### 4. Instalar packages necessárias
    sudo apt install python3-venv  
    sudo apt install libportaudio2  
    sudo apt install libatlas-base-dev

##### 5. Instalar as drivers do HAT
    git clone https://github.com/respeaker/seeed-voicecard.git  
    cd seeed-voicecard  
    sudo ./install.sh --compat-kernel  
    sudo reboot  
    arecord -L # Verificar se ficou bem instalado  

##### 6. Instalar o docker
    curl -fsSL https://get.docker.com -o get-docker.sh  
    sudo sh get-docker.sh

##### 7. Instalar o openHAB
    groupadd -g 9001 openhab  
    useradd -g 9001 openhab  
    usermod -a -G openhab pi # Substituir pi com user da sua escolha  

Para mudar a port usar -p (ex.: -p 8081), o defeito é 8080.  
Por defeito o openHAB é corrido ao restart, mas pode ser mudado com a --restart e um valor diferente  
    
    docker run \
           --name openhab \
           --net=host \
           --tty \
           -v /etc/localtime:/etc/localtime:ro \
           -v /etc/timezone:/etc/timezone:ro 
           -v openhab_addons:/openhab/addons \
           -v openhab_conf:/openhab/conf \
           -v openhab_userdata:/openhab/userdata \
           -d \
           --restart=always \
           openhab/openhab:3.0.2
   
##### 8. Configurar openHAB
Seguir as configurações no [README.md](https://github.com/hexoance/LPI_PROJ/blob/master/README.md) referentes ao openHAB.


##### 9. Testar o microfone (opcional)
    mkdir python-audio-example
    cd ./python-audio-example
    wget https://raw.githubusercontent.com/spatialaudio/python-sounddevice/0.4.1/examples/wire.py
    python3 -m venv ./venv
    source ./venv/bin/activate
    pip install sounddevice
    pip install numpy
    python3 wire.py # CTRL+C para terminar processo
    deactivate

##### 10. Instalar Edge Device
    git clone https://github.com/hexoance/LPI_PROJ.git
    cd LPI_PROJ/edge_device
    python3 -m venv ./venv
    source ./venv/bin/activate
    pip install python-openhab
    pip install numpy
    pip install sounddevice
    pip install --index-url=https://google-coral.github.io/py-repo/ tflite-runtime
    pip install pydub
    python3 main.py # CTRL+C para terminar processo
    deactivate
