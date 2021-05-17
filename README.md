# 'Smart Home' - Laboratório de Projeto Integrado

## Descrição:

Detetar e classificar atividades no âmbito doméstico através da captação de som e imagens em tempo real. Utiliza ainda
vários dispositivos inteligentes com sensores (_edge devices_) e auxilia-se de redes neurais para processar a informação
obtida.

## Como utilizar?

### OpenHAB3

1. **Configurações**
2. **Items**
    1. Add Item: Name = EdgeDeviceData
    2. Add Item: Name = BrainDevice

3. **Rules**
    1. Create Rule: Name = EventHandlerAudio
        1. When... EdgeDeviceData changed
        2. Then... execute a give script (Rule DSL):
    

> val action = EdgeDeviceData.state
>
> sendHttpPostRequest('http://127.0.0.1:5000/action', 'application/json', '{"action":"' + action + '"}')

4. **Pages** 
    1. Overview: Add Block → Add Row → Add Column → Add Label → Configure Widget:
        1. Title: [TÍTULO]
        1. Label: [SELECIONAR_ITEM]
    
### Python / PyCharm
 
- Executar Ficheiro...

---

## Exemplos:

1. Anomalias (torneira aberta, fuga de água);
2. Atividades (lavar a loiça, cozinhar);
3. Acidentes (quedas);
5. Entre outros.

---

## Contribuintes:

* Jorge Lopes (38607)
* Luís Mota (38186)
* Sérgio Moita (38148)

