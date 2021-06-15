# 'Reconhecimento de Atividades em Espaços Inteligentes' - Laboratório de Projeto Integrado

## Descrição:

Detetar e classificar atividades no âmbito doméstico através da captação de som e imagens em tempo real. Utiliza ainda
vários dispositivos inteligentes com sensores (_edge devices_) e auxilia-se de redes neurais para processar a informação
obtida.

## Como utilizar?

### Raspberry Pi 4
As instruções para instalar o software necessário numa RPI4 encontram-se no ficheiro [INSTALL_RPI](https://github.com/hexoance/LPI_PROJ/blob/master/INSTALL_RPI.md).

### OpenHAB3

1. Fazer [Download e Instalação](https://www.openhab.org/download/) do openHAB.

2. Configurar openHAB para o funcionamento correto.
    1. Adicionar itens para guardar/apresentar informação dos edge devices com os nomes EdgeDeviceData e BrainDevice, em Configurações → Items → Adicionar Item
        
    2. Adicionar script/regra em: Configurações → Regras → Criar Regra
        
        1. When... EdgeDeviceData changed
        2. Then... execute a give script (Rule DSL):
            >  val action = EdgeDeviceData.state
            >
            > sendHttpPostRequest('http://127.0.0.1:5000/action', 'application/json', '{"action":"' + action + '"}')

    3. Configurar apresentação de informações (interfaces) em: Configurações → Páginas → Overview 
        1. Overview: Add Block → Add Row → Add Column → Add Label → Configure Widget:
            1. Title: Brain Device
            2. Label: BrainDevice
    
### Para Desenvolvedores

- Usar Python 3.8.
- Instalar todas as dependências necessárias no projeto (ex.: python-openhab, sounddevice, tensorflow, etc.).
- A única dependência excecional é 'tflite-runtime' porque se encontra num repositório diferente ao defeito do Python, por exemplo, no PyCharm é necessário adicionar a seguinte configuração:
    - File → Settings → Project Interpreter → Install → Manage Repositories → Add ' https://google-coral.github.io/py-repo/ '
- Os datasets não estão incluídos no repositório GitHub por possuirem um tamanho bastante elevado. Contudo, o seu download pode ser feito em [FSD50k](https://zenodo.org/record/4060432) e [ESC-50](https://github.com/karolpiczak/ESC-50). Devem ser colocados na pasta do projeto, em: retrain-models/datasets. Se o disco do projeto não for suficiente para guardar os datasets, o caminho para os mesmos, pode ser passado como argumento para os ficheiros de retreino.

## Contribuintes:

* Jorge Lopes (38607)
* Luís Mota (38186)
* Sérgio Moita (38148)

