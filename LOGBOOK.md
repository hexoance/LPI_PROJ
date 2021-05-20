 ## **18/02/2021**
#### Contextualização do Projeto:
- ['Dive into Deep Learning' book](https://d2l.ai/)
- [Sound Classification (Generic)](https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7)
- [Sound Classification (Birds)](https://www.edgeimpulse.com/blog/bird-classification-lacuna-space)
- [Google Public Datasets](https://research.google/tools/datasets/)

#### Investigar o uso das seguintes ferramentas:
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [OpenHab](https://www.openhab.org/)

## **25/02/2021**
#### Por onde começar:

- Som:
  - Utilizar um modelo já feito e colocar a funcionar (Exemplo: Som dos pássaros)
   Posteriormente, introduzir conceitos nossos (ex. agua a correr, chuveiro, torneira, etc).
- Visão:
  - Analisar o trabalho do Aguiar

- Fazer o nosso próprio Binding consoante os sensores no openHab
  - Exemplo de estrutura: 
    - HTTP Binding (A mensagem em json deverá conter: hora, atividade e a accuracy, por exemplo).

## **04/03/2021**

#### Definir alguns Casos de Uso:
- Detetar acontecimentos passiveis de emergência (queda de alguma pessoa).
- Deduzir um horário dos ocupantes da casa (ex. não está em casa das 9h as 17h).
  - Reportar um alerta se for detetada uma presença nessa altura (ex. possível ladrão).
- Detetar uma fuga de água (ex. se for detetada torneira aberta mais do que 1h).
- Apagar a luz da divisão se não for detetado nenhum movimento.

Exemplo:
Deteto que a torneira está aberta e a pessoa sai da divisão (Poderá ser um alerta preventivo)
Se continuar, aumentar o grau do alerta.

- Criar o nosso próprio modelo com apenas as classes necessárias (casos de uso) ou 
  re-treinar um modelo já existente.
  - [Model Retrainig Example](https://www.tensorflow.org/hub/tutorials/tf2_image_retraining) 
  - Observações:
    - _model_handel_ <=> yamnet
    - (...) _.Dense_ ( novo numero de classes)

- Novo Dataset: [FreeSound50K](https://zenodo.org/record/4060432) (200 Classes)

## **11/03/2021**

#### Discussão de diagramas de workflow do projeto:
![LPI](assets/logbook_imgs/LPI.jpeg)

![LPI_EdgeDevice](assets/logbook_imgs/LPI_(EdgeDevice).jpeg)

- Limitar as classes para apenas os casos de uso a desenvolver.

## **18/03/2021**

- Testar com ficheiros audio gravados e testar.
- Usar code ufp pt para partilhar codigo com os orientadores do projeto.
- Tratamento/processamento de dados é necessário (Fazer step-by-step através da gravação para ficheiro).
- Usar a biblioteca mais tarde PyAudio para detetar diretamente no microfone o audio.
- Testar conjuntos para teste e conjuntos para treino.

## **25/03/2021**

- Balancear os ficheiros de audio na leitura (mesmo número de ficheiros para cada classe)
- [LSTE](https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/#heading3)

- Em relação ao video:
  - Usar Video para identificar: a pessoa "interessante" (ex. usar o rosto), a localização na casa (ex. quarto, cozinha, etc), a ação dessa pessoa (ex. sentada, cozinhar)

## **15/04/2021**
- Trabalhar em condições ótimas (sem ruídos e sem silêncio):
   - Ex: 2 Classes => 40 Sons perfeitos
  

- Passar o som com ruído por um filtro para deixar o som em especifico sozinho:
   - Ex. Computer_keyboard + Ruido => Filtro => Computer_keyboard


- Fazer Excel com sequencias de audio significativas nos ficheiros.

## **22/04/2021**

- Usar apenas o validation com as percentagens 80% treino e 20% teste.
- Retirar silêncio apenas no inicio e no fim dos ficheiros audio (.WAV).
- Analisar os ficheiros um a um do dataset (Polir o dataset).
- Adicionar ou não mais dados de outro dataset.
       
## **29/04/2021**

- Criar e implementar uma nova classe: Silêncio.
- Verificar se o silêncio existente no ficheiro é maior do que o chunk de procura, de forma a não estragar o padrão caraterístico do som.
- Fazer o pré-processamento do áudio (raw data) detetado no microfone e só depois passá-lo ao modelo.
- Gravam se detetarem movimento/audio.

## **06/05/2021**

- Enviar para openhab atividades reconhecidas pelo modelo
   - Pegar nessa atividade guarda-la com timestamp
   - Trabalhar essa informação (MODA?)
- No relatório colocar TUDO: o que foi feito, o que não foi feito, o que gostaríamos de ter feito, etc.
   - Diagramas, fluxogramas. Nada de código!

## **13/05/2021**

- Matriz de confusão multiclasse:
   - Qual o erro sistemático que o código está a ter. 
   - [Documentação TensorFlow](https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix)
   - [Exemplo GeeksForGeeks](https://www.geeksforgeeks.org/python-tensorflow-math-confusion_matrix/)
- Esclarecimento de dúvidas relativamente à troca de mensagens entre o Edge Device, openHAB e Brain.
- [GitHub Gist - Code Snippets](https://gist.github.com/)
- OpenCV trocar FPS usar função waitKey().
- Relatório: começar pela estrutura, headings e subheadings, depois inserir esboços do que se vai falar


## **20/05/2021**

- Matriz de Confusão deve ser feita apenas para o _validation set_.
- No relatório, abordar o tema da captação espacial do _array_ de microfones.
- Documentar testes de ligação entre o openHAB e o _Brain Device_.
