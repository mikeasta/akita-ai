<img src="https://cdn.dribbble.com/users/1885550/screenshots/3795640/akita-inu.jpg" align="right" width="300px"></img>
<h1>Akita.AI - Рекомендательная система</h1>

<h3><b>Систему разработали студенты группы 0391:</b></h3>
<ul> 
  <li>Андрющенко Ксения</li>
  <li>Асташёнок Михаил</li>
</ul>
<br>

<h3>Введение</h3>
<p>Система разработана в рамках Альтернативного Экзамена по дискретной математике 2021 по теме "Рекомендательные системы". Данный проект включает в себя два реализованых алгоритма работы систем основанных на коллаборативной фильтрации и на контенте. Ознакомиться с теоретическим материалом по теме "Рекомендательные системы" можно с документами "РС: Теория" и "РС: Презентация", находящимися в директории с материалами нашей команды.</p>
<br>

<h3>Краткая характеристика содержимого репозитория:</h3>
<ul> 
  <li><b>collaborative.py</b> - код реализованной РС, основанной на коллаборативной фильтрации.</li>
  <li><b>content-based.py</b> - код реализованной РС, основанной на контенте.</li>
  <li><b>database.json</b> - кастомная база данных для демонстрации работы системы.</li>
  <li><b>json_implementation.py</b> - небольшой самодельный модуль для взаимодействия с JSON базой данных.</li>
  <li><b>config.py</b> - конфигурационный файл. Несмотря на то, что содержит лишь одну переменную, при редактировании и увеличении сложности проекта, такие конфигурационные файлы становятся крайне полезны.</li>
</ul>
<p>Файлы <b>.gitignore</b> и <b>README.md</b> служат для облегчения разработки и документирования программы. Они никак не относятся к работе программы</p>
<br>

<h3>Collaborative.py - коллаборативная фильтрация </h3>
<p>Вспомним, каким образом происходит коллаборативная фильтрация в рекомендательной системе, основанной на одноименном методе. При этом рекомендательном методе мы не интересуемся самим продуктом, его содержимым настолько, насколько интересуемся оценками этого продукта другими пользователями, чьи интересы схожи с нашими. Данные о пользователе мы представляем в виде вектора оценок, которые он поставил тому или иному предмету. Количество координат вектора равно количеству имеющихся в "ассортименте" наименований предметов. Для рассчета близости интересов пользователей воспользуемся формулой рассчета <a href="https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F_%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C#%D0%9A%D0%BE%D1%81%D0%B8%D0%BD%D1%83%D1%81%D0%BD%D0%BE%D0%B5_%D1%81%D1%85%D0%BE%D0%B4%D1%81%D1%82%D0%B2%D0%BE">косинусного сходства</a>. Напомню, что чем ближе косинусное сходство к единице, тем более похожи друг на друга вектора.</p>
<br>

<h4> Перейдем к коду: </h4>
<p>Для начала импортируем необходимые модули: <b>numpy</b> для работы с векторами и <b>json_implementation</b> для загрузки данных</p>

```python
import numpy as np
import json_implementation as ji
```

<p>Загрузим массив векторов уже заразервированных пользователей:</p>

```python
ratings = ji.json_get_data()["collaborative_filtering_ratings"]
```

<p>Определим вектор пользователя, для которого будем искать косинусного сходства с другими пользователями:</p>

```python
handling_user_ratings = [1, 3, 0, 0, 0]
```

<p>Теперь определим функцию косинусного сходства <b>cosine(ratings_a, ratings_b)</b></p>

```python
def cosine(ratings_a, ratings_b):
    length_a = np.linalg.norm(ratings_a)
    length_b = np.linalg.norm(ratings_b)

    distance = np.dot(ratings_a, ratings_b) / np.dot(length_a, length_b)
    return distance
```

<p> На вход данной функции принимается два вектора пользовательских оценок, между которыми мы и будем искать косинусное сходство. С помощью метода <b>np.linalg.norm()</b> запишем в переменные <b>length_*</b> длины векторов. Косинусное расстояние <b>distance</b> вычислим как частное скалярного произведения векторов и произведения длин тех же векторов.</p>

<p>Теперь проведем рассчет и сравним вектор нашего заданного пользователя со всеми заразервированными пользователями:</p>

```python
cosine_values = []
for user_ratings in ratings:
    current_distance = cosine(handling_user_ratings, user_ratings)
    cosine_values.append({current_distance: user_ratings})
   
cosine_values.sort(key=lambda item: sorted(list(item.keys())), reverse=True)
print(cosine_values)
```

<p> Создали пустой массив <b>cosine_values</b>, который будет хранить в себе объекты : "вектор" : "косинусное сходство с заданным пользователем". Цикл перебирает каждый вектор в массиве векторов, сохраняет в переменную <b>current_distance</b> результат подсчета косинусного сходства заданного и итерируемого пользователя и записывает в массив новый объект. Методом массивов <b>.sort()</b> сортируем в обратном порядке массив косинусных расстояний - ведь чем больше косинусное расстояние , тем более похожий на заданного пользователя итерируемый пользователь.</p>
<p>Запускаем .py документ и наблюдаем: </p>

```
[{0.9534625892455924: [1, 3, 0, 1, 0]}, {0.8280786712108251: [0, 4, 1, 0, 2]}, {0.7911548052852398: [1, 4, 3, 0, 1]}, {0.760638829255665: [3, 2, 0, 1, 0]}, {0.5976143046671968: [1, 3, 0, 3, 3]}, {0.50709255283711: [3, 1, 0, 0, 2]}, {0.42426406871192845: [0, 1, 0, 2, 0]}, {0.35233213170882205: [0, 2, 3, 0, 4]}, {0.3450327796711771: [2, 1, 0, 4, 0]}, {0.29814239699997197: [4, 0, 1, 1, 0]}, {0.28603877677367767: [0, 1, 0, 3, 1]}, {0.28284271247461895: [2, 0, 0, 0, 1]}, {0.2300894966542111: [0, 1, 4, 0, 0]}, {0.07254762501100116: [1, 0, 3, 3, 0]}, {0.0: [0, 0, 1, 3, 4]}, {0.0: [0, 0, 3, 1, 2]}]
```

<p>Как можем заметить, первый вектор <b>[1, 3, 0, 1, 0]</b> система принимает за самый похожий на наш заданный <b>[1, 3, 0, 0, 0]</b>. И ведь это именно так! Исходя из таких сравнений система начинает понимать, стоит ли по оценкам похожих пользователей предлагать данному пользователю данный продукт.</p>
<br>
<h3>Content-based.py - фильтрация на основе контента </h3>
<p>Данный тип рекомендательных систем немного сложнее, нежели коллаборативная фильтрация, однако познакомившись с принципом работы косинусного сходства, нам будет намного проще разобрать систему, основанную на контенте. Здесь читателю необходимо изучить <a href="https://ru.wikipedia.org/wiki/TF-IDF">TF-IDF меру</a> для того, чтобы разобраться с принципом работы следующего кода. Фильтрация на основе контента уже не столько относится к сравнению между собой интересов пользователей, сколько сравнение интересов пользователя с контентом, имеющимся на сайте или в приложении. Как и при коллаборативной фильтрации мы будем иметь дело с векторами, только в данном случае - векторами весов слов, содержащихся в тексте.</p>
<p>Формула рассчета весов слов: <i>w<sub>x,y</sub> = tf<sub>x,y</sub> * log⁡(N / df<sub>x</sub>)</i></p>
<p>Где: <p>
<ul>
  <li><i>tf<sub>x,y</sub></i> -  частота слова х в описании товара у</li>
  <li><i>df<sub>x</sub></i> - количество товаров, содержащих в своем описании слово х</li>
  <li><i>N</i> - общее количество товаров</li>
</ul>
<p>Алгоритм в двух словах таков: мы разбиваем каждое описание по словам, считаем их веса, записываем их в вектора и сравниваем эти вектора друг с другом.</p>
<br>

<h4> Перейдем к коду: </h4>
<p>Для начала импортируем нужные нам библиотеки:</p>

```python
import numpy as np
import json_implementation as ji
import string
```

<p>Все то же самое, что и в <b>collaborative.py</b>, только добавился <b>import string</b>. Он нам понадобится при разделении текста на слова. </p>
<p>Дальше сохраняем в переменную <b>texts</b> тексты наших зарезервированных описаний, <b>document_amount</b> - длину коллекции описаний (количество текстов описаний),а в <b>current_user_text</b> - текст описания предмета, который понравился нашему пользователю (на этот текст мы и будем ориентироваться при подборе подходящего предмета).</p>
  
 ```python
texts = ji.json_get_data()["content_based_texts"]
document_amount = len(texts)

current_user_text = texts[0]
```

<p> Дальше прописываем уже знакомую нам функцию рассчета косинусного сходства: </p>

```python
def cosine(vector_a, vector_b):
    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)

    return np.dot(vector_a, vector_b) / np.dot(length_a, length_b)
```

<p>В качестве следующей функции реализуем подсчет веса слов в тексте и получение нужного нам объекта весов:</p>

```python
def calc_words_weight(text):
    word_array = [word.strip(string.punctuation).lower() for word in text.split()]

    word_vocabulary = {}
    for word in word_array:
        word_vocabulary[word] = word_vocabulary[word] + 1 if word in word_vocabulary else 1
        
    unique_words_amount = len(word_vocabulary)
        
    for word in word_vocabulary:
        # IDF value 
        found_times = 0
        for i in range(document_amount):
            if texts[i].lower().find(word) != -1:
                found_times += 1

        idf = np.log10(document_amount / found_times)

        # TF value
        tf = word_vocabulary[word] / unique_words_amount

        # TF-IDF value
        word_vocabulary[word] = tf * idf
    
    return word_vocabulary
```

<p>Функция <b>calc_words_weight(text)</b> принимает в качестве аргумента текст и возвращает объект весов каждого слова. Разберем эту функцию построчно.</p>

<p>Cперва мы разобьем наш текст на массив отдельных слов с помощью данной строки кода:</p>

```python
word_array = [word.strip(string.punctuation).lower() for word in text.split()]
```

<p>Стоит отметить, что данный массив может содержать в себе копии слов, поэтому нужно будет позаботиться о создании переменной, хранящей в себе эти слова в единичном экземпляре и количество использований слов в данном тексте.</p>
<p>Переменная <b>word_vocabulary</b> и послужит нам таким хранилищем. Это объект вида: "слово" : "количество_его_вхождений_в_описание"</p>

```python
word_vocabulary = {}
for word in word_array:
    word_vocabulary[word] = word_vocabulary[word] + 1 if word in word_vocabulary else 1
```

<p>Цикл проходится по каждому слову в массиве <b>word_array</b> и сохраняет его в качестве ключа свойства переменной <b>word_vocabulary</b>. Если итерируемое слово уже содержится в качестве ключа одного из свойств <b>word_vocabulary</b>, то численное значение этого свойства (то бишь количество итерируемых слов в данном тексте) увеличивается на один. Иначе, если слово до этого еще не встречалось, соответствующему свойству присвоим единицу.</p>
<p>Далее сохраним в качестве переменной количество уникальных слов в тексте: </p>

```python
unique_words_amount = len(word_vocabulary)
```

<p><b>unique_words_amount</b> нам пригодится, когда мы будем рассчитывать <b>TF</b>-коэффициент данного слова.</p>
<p>Далее разберем цикл, рассчитывающий для каждого слова, его вес во всей базе данных текстов описаний.</p>

```python
for word in word_vocabulary:
    # IDF value 
    found_times = 0
    for i in range(document_amount):
        if texts[i].lower().find(word) != -1:
            found_times += 1

     idf = np.log10(document_amount / found_times)

    # TF value
    tf = word_vocabulary[word] / unique_words_amount

    # TF-IDF value
    word_vocabulary[word] = tf * idf
```

<p>Данный цикл можно разбить на три блока, отмеченных комментариями. Первый блок рассчитывает <b>IDF</b>-коэффициент. Внутренний цикл считает, сколько текстов в базе данных содержат итерируемое слово. Это значение необходимо для рассчета <b>IDF</b>-коэффициента. Формула для рассчета этого коэффициента: <b>log(количество_описаний_в_базе_данных / количество_описаний_содержащих_в_себе_итерируемое слово)</b></p>
<p>Следующий блок рассчитывает <b>TF</b>-коэффициент данного слова. Как раз тут нам и пригодится переменная <b>unique_words_amount</b>.</p>
<p>В конце цикла мы рассчитываем <b>TF-IDF</b>-коэффициент - произведение <b>TF</b> и <b>IDF</b> коэффициентов.</p>
<p>В конце работы возвращаем объект уникальных слов, содержащихся в данном тексте, и их весов в базе данных</p>

```python
return word_vocabulary
```

<p>Последним штрихом в работе данной системы послужит функция <b>calc_cosine_distances(text_to_compare)</b>, которая вернет для каждого текста описания косинусное сходство с выбранным нами ранее текстом описания.</p>

```python
def calc_cosine_distances(text_to_compare):
    cosine_distances = []

    for i in range(document_amount):
        comparable_text_library = calc_words_weight(text_to_compare)
        current_text_library    = calc_words_weight(texts[i])

        for word in comparable_text_library:
            if word not in current_text_library:
                current_text_library[word] = 0

        for word in current_text_library:
            if word not in comparable_text_library:
                comparable_text_library[word] = 0

        vector_comparable = [comparable_text_library[word] for word in comparable_text_library]
        vector_current    = [current_text_library[word]    for word in comparable_text_library]

        cosine_distances.append(cosine(vector_comparable, vector_current))
    
    return cosine_distances
```

<p>На вход функция принимает текст описания, с которым мы должны сравнить все остальные тексты описаний в базе данных. Функция возвращает массив косинусных сходств выбранного текста со всеми остальными.</p>
<p>Разберем построчно:</p>

```python
cosine_distances = []
```

<p>Переменная-массив, которая будет хранить в себе косинусные расстояния. Этот же массив мы заполним в цикле и в конце работы функции вернем.</p>
<p>Перейдем к разбору цикла:</p>

```python
for i in range(document_amount):
        comparable_text_library = calc_words_weight(text_to_compare)
        current_text_library    = calc_words_weight(texts[i])

        for word in comparable_text_library:
            if word not in current_text_library:
                current_text_library[word] = 0

        for word in current_text_library:
            if word not in comparable_text_library:
                comparable_text_library[word] = 0

        vector_comparable = [comparable_text_library[word] for word in comparable_text_library]
        vector_current    = [current_text_library[word]    for word in comparable_text_library]

        cosine_distances.append(cosine(vector_comparable, vector_current))
```

<p>1. Переменные <b>comparable_text_library</b> и <b>current_text_library</b> хранят в себе словари уникальных слов с их весами: <b>comparable_text_library</b> - описания, переданного в качестве аргумента в функцию <b>calc_cosine_distances()</b>, <b>current_text_library</b> - итерируемого описания.</p>
<p>Следующим шагом мы "сбалансируем" оба словаря: если слово есть в одном словаре, но отсутствует во втором, то во втором словаре свойство по ключу этого слова приобретает значение 0. Балансировка производится следующими двумя циклами:</p>

```python
for word in comparable_text_library:
    if word not in current_text_library:
        current_text_library[word] = 0

for word in current_text_library:
    if word not in comparable_text_library:
        comparable_text_library[word] = 0
```

<p>Для чего нам нужна была балансировка выше? Она нам необходима для создания векторов весов слов сравниваемых описаний. Для того, чтобы поставить все необходимые слова в вектор, мы и должны перебирать, проще говоря, общий словарь всех вместе взятых слов в сравниваемых описаниях. А для того, чтобы задать нужную позицию нужной координате вектора, будем перебирать слова строго по одному и тому же словарю (в нашем случае по словарю <b>comparable_text_library</b>):</p>

```python
vector_comparable = [comparable_text_library[word] for word in comparable_text_library]
vector_current    = [current_text_library[word]    for word in comparable_text_library]
```

<p>В конце каждого цикла рассчитаем косинусное сходство двух векторов, заполненных выше и добавим данное расстояние в массив косинусных расстояний. Вернем данный массив:</p>

```python
return cosine_distances
```

<p>Рассчитаем все расстояния и выведем в консоль результат фильтрации базы данных:</p>

```python
cosine_distance = calc_cosine_distances(current_user_text)
print(cosine_distance)
```

<p>Видим в консоли:</p>
```
[1.0, 0.030152426954748984, 0.013230729574516273, 0.022993496297877468, 0.0025800777912773722, 0.04655763636782153, 0.01640286400535773, 0.02678171870008438, 0.022821365099150722, 0.0317724470755004, 0.047416344325109175, 0.026850780497314753, 0.001052066339123618, 0.006789040882787158, 0.020286825521895646, 0.017898539423519832]
```

<p>В консоль вывелись все косинусные сравнения самого первого описания(которого мы выбрали в качестве отправной точки для сравнения) со всеми остальными текстами. Как мы можем заметить, первое значение равно единице. Такой результат и был нужен, ведь мы сравниваем одинаковые тексты.</p>
