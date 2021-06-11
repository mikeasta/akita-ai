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
<p>Для начала импортируем необходимые модули: <b>numpy</b> для работы с векторами и <b>json_implementation</b> для загрузки данных. Также импортируем переменные <b>RATING_TO_RECOMMEND, SAME_RATIO</b> из файла конфигурации. <b>RATING_TO_RECOMMEND</b> показывает, ниже какой оценки нельзя рекомендовать пользователю товар. <b>SAME_RATIO</b> - точность, которую мы требуем, чтобы сравниваемый вектор имел, чтобы мы могли спокойно по нему сделать предсказание:</p>

```python
import numpy as np
from config import RATING_TO_RECOMMEND, SAME_RATIO
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

<p> Создали пустой массив <b>cosine_values</b>, который будет хранить в себе объекты : "косинусное сходство с заданным пользователем" : "вектор". Цикл перебирает каждый вектор в массиве векторов, сохраняет в переменную <b>current_distance</b> результат подсчета косинусного сходства заданного и итерируемого пользователя и записывает в массив новый объект. Методом массивов <b>.sort()</b> сортируем в обратном порядке массив косинусных расстояний - ведь чем больше косинусное расстояние, тем более похожий на заданного пользователя итерируемый пользователь.</p>
<p>Осталось изучить последнюю функцию, которая собственно и предскажет, какие оценки поставит пользователь товарам:</p>

```python
# Recommendation UI
def recommend(cosine_values, user_ratings):

    # Check, if we need to make a prediction
    if 0 not in user_ratings: return

    # Define, which goods we should recommend or not
    print("\nUser ratings:", user_ratings)

    # Buffer rating
    buffer_user_rating = list.copy(user_ratings)

    # Handle each vector to predict users rating
    for user_index, user_data in enumerate(cosine_values):
        
        # Get it's key & property
        key_distance = list(user_data.keys())[0]
        prop_vector  = user_data[key_distance]
        print("Vector", user_index + 1, ":", key_distance, ", ", prop_vector)

        # Check if difference is too big
        if key_distance < SAME_RATIO: break

        # Handle every coordinate in prop vector
        for i in range(5):
            # Check if data and current prop vector coordinate is already written
            if prop_vector[i] != 0 and buffer_user_rating[i] == 0:
                buffer_user_rating[i] = prop_vector[i]
    
        if 0 not in buffer_user_rating: break

    print("Most nearest prediction:", buffer_user_rating, "\n")

    # Say, if we recommend this good to recommend 
    for i in range(5):
        # Check, if rating is already written
        if user_ratings[i] != 0: continue

        recommend_string = ""
        if buffer_user_rating[i] == 0:
            recommend_string = "we can't make a reliable prediction with that good"
        elif buffer_user_rating[i] < RATING_TO_RECOMMEND:
            recommend_string = "we don't recommend this good"
        else: 
            recommend_string = "we recommend this good"

        print("Good", i + 1, ":", recommend_string)
```
<p>На вход функция принимает два параметра: найденные косинусные расстояния и сам вектор пользователя. Смысл функции весьма прост: предсказать, какие оценки поставит товарам пользователь, исходя из оценок схожих по интересам других пользователей, и определить по найденным оценкам, стоит ли вообще рекомендовать пользователю данный продукт, или все таки не стоит из-за низких оценок. Разберем функцию построчно:</p>

<p>Проверим, существует ли вообще необходимость в предсказании. Ведь если все оценки уже выставлены, то смысла предсказывать, очевидно, нет. Эта проверка окажется бесполезной, т.к. в ситуации, когда наименований товаров насчитывается несколько сотен или даже тысяч, пользователь очень навряд ли смог бы проставить абсолютно все оценки всем товарам. Поэтому в большом проекте эта проверка ненужна, но так как мы имеем дело с очень ограниченной библиотекой товаров, такая проверка не будет лишней.</p>

```python
# Check, if we need to make a prediction
if 0 not in user_ratings: return
```

<p>Выводим в консоль имеющийся массив словарей расстояний до векторов и копируем в буферный массив оценки нашего пользователя. Он нам пригодится для заполнения нашего предсказания.</p>

```python
# Define, which goods we should recommend or not
print("\nUser ratings:", user_ratings)

# Buffer rating
buffer_user_rating = list.copy(user_ratings)
```
<p>Затем нам собственно и нужно заполнить наш буферный вектор для предсказания. С помощью цикла мы перебираем каждый похожий вектор (каждый новый вектор похож на нашего пользователя все меньше), пока мы не дойдем до вектора, уровень схожести с которым опустится ниже заданного нами SAME_RATIO или пока мы не совершим все предсказания и полностью не заполним наш буферный вектор. Построчный разбор чуть ниже:</p>

```python
# Handle each vector to predict users rating
for user_index, user_data in enumerate(cosine_values):
    
    # Get it's key & property
    key_distance = list(user_data.keys())[0]
    prop_vector  = user_data[key_distance]
    print("Vector", user_index + 1, ":", key_distance, ", ", prop_vector)

    # Check if difference is too big
    if key_distance < SAME_RATIO: break

    # Handle every coordinate in prop vector
    for i in range(5):
        # Check if data and current prop vector coordinate is already written
        if prop_vector[i] != 0 and buffer_user_rating[i] == 0:
            buffer_user_rating[i] = prop_vector[i]

    if 0 not in buffer_user_rating: break
```
<p>Сохраним данные ключа и вектора по этому ключу в переменные, а также выведем в консоль итерируемый вектор.</p>

```python
# Get it's key & property
key_distance = list(user_data.keys())[0]
prop_vector  = user_data[key_distance]
print("Vector", user_index + 1, ":", key_distance, ", ", prop_vector)
```
<p>Запишем условие, что если мы достигли границы схожести, цикл прекращает работу.</p>

```python
# Check if difference is too big
if key_distance < SAME_RATIO: break
```
<p>Циклом переберем итерируемый вектор: если у итерируемого вектора в позиции i есть значение, а у буферного - нет, тогда сохраним в буферный вектор то значение, отличное от нуля, которое мы нашли у итерируемого вектора.</p>

```python
# Handle every coordinate in prop vector
for i in range(5):
    # Check if data and current prop vector coordinate is already written
    if prop_vector[i] != 0 and buffer_user_rating[i] == 0:
        buffer_user_rating[i] = prop_vector[i]
```

<p>В конце цикла сделаем проверку на наличие в буферном векторе нулевых значений.</p>

```python
if 0 not in buffer_user_rating: break
```

<p>В функции <b>recommend()</b> выведем в консоль получившийся предсказанный вектор</p>

```python
print("Most nearest prediction:", buffer_user_rating, "\n")
```

<p>В последнем цикле функции мы будем сравнивать предсказанный вектор с нашим исходным, найдем новые вставленные предсказанные оценки, а заодно скажем, стоит ли рекомендовать данный товар пользователю или нет:</p>

```python
# Say, if we recommend this good to recommend 
for i in range(5):
    # Check, if rating is already written
    if user_ratings[i] != 0: continue

    recommend_string = ""
    if buffer_user_rating[i] == 0:
        recommend_string = "we can't make a reliable prediction with that good"
    elif buffer_user_rating[i] < RATING_TO_RECOMMEND:
        recommend_string = "we don't recommend this good"
    else: 
        recommend_string = "we recommend this good"

    print("Good", i + 1, ":", recommend_string)
```

<p>Вызовем <b>recommend()</b> с надлежащими аргументами:</p>

```python
recommend(cosine_values, handling_user_ratings)
```

```
[{0.9534625892455924: [1, 3, 0, 1, 0]}, {0.7911548052852398: [1, 4, 3, 0, 1]}, {0.760638829255665: [3, 2, 0, 1, 0]}, {0.6605782590758164: [0, 4, 1, 0, 4]}, {0.5976143046671968: [1, 3, 0, 3, 3]}, {0.50709255283711: [3, 1, 0, 0, 2]}, {0.42426406871192845: [0, 1, 0, 2, 0]}, {0.35233213170882205: [0, 2, 3, 0, 4]}, {0.3450327796711771: [2, 1, 0, 4, 0]}, {0.29814239699997197: [4, 0, 1, 1, 0]}, {0.28603877677367767: [0, 1, 0, 3, 1]}, {0.28284271247461895: [2, 0, 0, 0, 1]}, {0.2300894966542111: [0, 1, 4, 0, 0]}, {0.07254762501100116: [1, 0, 3, 3, 0]}, {0.0: [0, 0, 1, 3, 4]}, {0.0: [0, 0, 3, 1, 2]}]

User ratings: [1, 3, 0, 0, 0]
Vector 1 : 0.9534625892455924 ,  [1, 3, 0, 1, 0]
Vector 2 : 0.7911548052852398 ,  [1, 4, 3, 0, 1]
Most nearest prediction: [1, 3, 3, 1, 1]

Good 3 : we don't recommend this good
Good 4 : we don't recommend this good
Good 5 : we don't recommend this good
```

<p>Как можем заметить, первый вектор <b>[1, 3, 0, 1, 0]</b> система принимает за самый похожий на наш заданный <b>[1, 3, 0, 0, 0]</b>. И ведь это именно так! Ниже программа провела осмотр векторов по заданной нами точности и предсказала, какие оценки поставит данный пользователь тем или иным товарам, а в конце подытожила, стоит ли рекомендовать данный товар, или нет.</p>
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
# Dependencies
import numpy as np
import json_implementation as ji
import string
import spacy
from stop_words import get_stop_words
```

<p>Стоит отметить, что в данной программе мы будем работать с текстами, поэтому нам важно импортировать нужные нам модули. <b>import string</b> на понадобится при разделении строки по знакам препинания и по пробелам, <b>import spacy</b> используется для <a href="https://ru.wikipedia.org/wiki/%D0%9B%D0%B5%D0%BC%D0%BC%D0%B0%D1%82%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F" target="_blank">лемматизации</a> текста, а <b>from stop_words import get_stop_words</b> для чистки предложений от так называемых стоп-слов (артикли, предлоги, союзы и т.д.).</p>
<p>Дальше сохраняем в переменную <b>texts</b> тексты наших зарезервированных описаний, <b>document_amount</b> - длину коллекции описаний (количество текстов описаний),а в <b>current_user_text</b> - текст описания предмета, который понравился нашему пользователю (на этот текст мы и будем ориентироваться при подборе подходящего предмета).</p>
  
```python
texts = ji.json_get_data()["content_based_texts"]
document_amount = len(texts)

current_user_text = texts[0]
```

<p> Инициализируем модель <b>nlp</b> пакета <b>spacy</b> для последующих лемматизаций текста.</p>

```python
nlp = spacy.load("en_core_web_sm")
```

<p> С использованием импортированной из <b>stop_words</b> функции <b>get_stop_words()</b> получим массив стоп-слов английского языка</p>

```python
stop_words_array = get_stop_words("en")
```

<p> Дальше прописываем функцию рассчета косинусного сходства: </p>

```python
def cosine(vector_a, vector_b):
    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)

    return np.dot(vector_a, vector_b) / np.dot(length_a, length_b)
```

<p>В качестве следующей функции реализуем подсчет веса слов в тексте и получение нужного нам объекта весов:</p>

```python
def calc_words_weight(text):

    # Lemmatizing
    word_array = nlp(text)
    text = " ".join([token.lemma_ for token in word_array])

    # Deleting stop words
    word_array = [word.strip(string.punctuation).lower() for word in text.split() if word.strip(string.punctuation).lower() not in stop_words_array]
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

        idf = np.log10(document_amount / found_times or 1)

        # TF value
        tf = word_vocabulary[word] / unique_words_amount

        # TF-IDF value
        word_vocabulary[word] = tf * idf
    
    return word_vocabulary
```

<p>Функция <b>calc_words_weight(text)</b> принимает в качестве аргумента текст и возвращает объект весов каждого слова. Разберем эту функцию построчно.</p>

<p>Сперва применим лемматизацию. Мы разбиваем текст по словам при помощи <b>nlp()</b>, а затем объединяем в один текст, но уже в текст вставляем леммы слов, на которые был разбит наше текст:</p>

```python
# Lemmatizing
word_array = nlp(text)
text = " ".join([token.lemma_ for token in word_array])
```

<p>Затем мы разобьем наш текст на массив отдельных слов и удалим все "стоп_слова" с помощью данной строки кода:</p>

```python
word_array = [word.strip(string.punctuation).lower() for word in text.split() if word.strip(string.punctuation).lower() not in stop_words_array]
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

     idf = np.log10(document_amount / found_times or 1)

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

<p>Переменные <b>comparable_text_library</b> и <b>current_text_library</b> хранят в себе словари уникальных слов с их весами: <b>comparable_text_library</b> - описания, переданного в качестве аргумента в функцию <b>calc_cosine_distances()</b>, <b>current_text_library</b> - итерируемого описания.</p>
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
[1.0000000000000002, 0.014574464017916072, 0.006950004010859684, 0.009343378186624956, 0.0, 0.02184086498889478, 0.008625111603308587, 0.00798807893226927, 0.01684748287865778, 0.03248307157847479, 0.021846308812417985, 0.012770563670994781, 0.0, 0.0033431703273154277, 0.01923449069694336, 0.015871561515082355]
```

<p>В консоль вывелись все косинусные сравнения самого первого описания(которого мы выбрали в качестве отправной точки для сравнения) со всеми остальными текстами. Как мы можем заметить, первое значение равно единице. Такой результат и был нужен, ведь мы сравниваем одинаковые тексты.</p>
<p>Давайте протестируем нашу систему еще чуток. В <b>database.json</b> после первого описания вставим то же самое описание, только удалим первые два слова:</p>

```json
"content_based_texts": [
        "The hydra is a simple creature. Less than half an inch long, its tubular body has a foot at one end and a mouth at the other. The foot clings to a surface underwater — a plant or a rock, perhaps — and the mouth, ringed with tentacles, ensnares passing water fleas. It does not have a brain, or even much of a nervous system. And yet, new research shows, it sleeps. Studies by a team in South Korea and Japan showed that the hydra periodically drops into a rest state that meets the essential criteria for sleep. On the face of it, that might seem improbable. For more than a century, researchers who study sleep have looked for its purpose and structure in the brain. They have explored sleep’s connections to memory and learning. They have numbered the neural circuits that push us down into oblivious slumber and pull us back out of it. They have recorded the telltale changes in brain waves that mark our passage through different stages of sleep and tried to understand what drives them. Mountains of research and people’s daily experience attest to human sleep’s connection to the brain.",
        "is a simple creature. Less than half an inch long, its tubular body has a foot at one end and a mouth at the other. The foot clings to a surface underwater — a plant or a rock, perhaps — and the mouth, ringed with tentacles, ensnares passing water fleas. It does not have a brain, or even much of a nervous system. And yet, new research shows, it sleeps. Studies by a team in South Korea and Japan showed that the hydra periodically drops into a rest state that meets the essential criteria for sleep. On the face of it, that might seem improbable. For more than a century, researchers who study sleep have looked for its purpose and structure in the brain. They have explored sleep’s connections to memory and learning. They have numbered the neural circuits that push us down into oblivious slumber and pull us back out of it. They have recorded the telltale changes in brain waves that mark our passage through different stages of sleep and tried to understand what drives them. Mountains of research and people’s daily experience attest to human sleep’s connection to the brain.",
        "In a pair of advisories, the Treasury’s Office of Foreign Assets Control and its Financial Crimes Enforcement Network warned that facilitators could be prosecuted even if they or the victims did not know that the hackers demanding the ransom were subject to U.S. sanctions. Ransomware works by encrypting computers, holding a company’s data hostage until a payment is made. Organizations have often ponied up ransoms to liberate their data. “It is a game changer,” said Alon Gal, chief technology officer of Hudson Rock, which works to head off ransomware attacks before they happen. Before, companies could decide whether or not to pay cybercriminals off, he said. Now that those decisions are being brought under government oversight 'we are going to see a much tougher handling of these incidents.'The Enforcement Network’s advisory also warned that cybersecurity firms may need to register as money services businesses if they help make ransomware payments. That would impose a new reporting requirement on a previously little-regulated corner of the cybersecurity industry. Ransomware has become an increasingly visible threat in the United States and abroad. Cybercriminals have long used the software to loot their victims. Some countries, notably North Korea, are also accused of deploying ransomware to earn cash.",
        "One of the best ways to find ways to improve performance, we’ve found, is to work closely with our customers. We love shared troubleshooting sessions with their own engineering teams to find and eliminate bottlenecks. In one of these, we found a discrepancy in upload speeds between Windows and Linux. We then worked with the Windows Core TCP team to triage the problem, find workarounds, and eventually fix it. The issue turned out to be in the interaction between the Windows TCP stack and firmware on the Network Interface Controllers (NIC) we use on our Edge servers. As a result, Microsoft improved the Windows implementation of the TCP RACK-TLP algorithm and its resilience to packet reordering. This is now fixed and available starting with Windows 10 build 21332 through the Windows Insider Program dev channel. How we got there is an interesting and instructive tale—we learned that Windows has some tracing tools beyond our Linux-centric experience, and working with Microsoft engineers led to both a quick workaround for our users and a long-term fix from Microsoft. Teamwork! We hope our story inspires others to collaborate across companies to improve their own users’ experiences.",
        "Every one of us at PlanetScale is deeply unsatisfied by the current state of “modern” databases. We’ve worked at companies like YouTube, Amazon, Facebook, DigitalOcean, and GitHub, always focused on solving the same problem — database scaling.We’ve seen the same sad pattern repeat year after year. Companies pick a stack on day 1 that optimizes for developer velocity to get that MVP out the door. After they see some success, years 3 or 4 are spent paying down immense technical debt — mostly due to that early database choice. There needs to be a better wayPlanetScale’s technologies have been the choice of the hyperscalers for years. We’ve seen some of the largest web services in the world make Vitess their database of choice, scaling beyond imaginationThe world is full of hosted databases that don’t have much more to offer. The state of the art isn’t really that good. At PlanetScale we hold the belief that databases should have power and flexibility that make development joyful, along with the confidence that you’ll never outgrow it.Today we are thrilled to announce PlanetScale — the database for developers.",
        "Australian Hyundai rally driver Brendan Reeves is the man behind the latest record, having driven a production-spec Nexo from the Essendon Fields in MelbournAfter 807km of efficiency-focused driving, Reeves arrived in Broken Hill with plenty range still showing on the vehicle’s trip computer.The journey then continued to Silverton, an outback town on the outskirts of Broken Hill, best known as the setting for 1980s post-apocalyptic action film Mad Max 2, and the car then travelled some 60km beyond before the Nexo’s hydrogen tank was depleted on the Wilangee road beyond Eldee Station.During the trip the Nexo consumed a total of 6.27kg of hydrogen, at a rate of 0.706kg/100km. It purified 449,100 litres of air on the journey – enough for 33 adults to breathe in a day.According to Hyundai, the trip took 13 hours and six minutes at an average speed of 66.9km/h. The Nexo’s low fuel warning first lit up at 686km, with over 200km of range left from that point.The fuel light started flashing after 796km, with 90km of real range remaining.A representative from the RACV was on hand to seal the Nexo’s tank at the start of the journey, and an NRMA representative confirmed the validity of the tank seal at the end.Commenting on the achievement Reeves, said, 'Being a rally driver, I’ve always wanted to achieve a world record, but I could never have guessed it would come about this way.",
        "Most people think of nerds as quiet, diffident people. In ordinary social situations they are — as quiet and diffident as the star quarterback would be if he found himself in the middle of a physics symposium. And for the same reason: they are fish out of water. But the apparent diffidence of nerds is an illusion due to the fact that when non-nerds observe them, it's usually in ordinary social situations. In fact some nerds are quite fiercThe fierce nerds are a small but interesting group. They are as a rule extremely competitive — more competitive, I'd say, than highly competitive non-nerds. Competition is more personal for them. Partly perhaps because they're not emotionally mature enough to distance themselves from it, but also because there's less randomness in the kinds of competition they engage in, and they are thus more justified in taking the results personally.Fierce nerds also tend to be somewhat overconfident, especially when young. It might seem like it would be a disadvantage to be mistaken about one's abilities, but empirically it isn't. Up to a point, confidence is a self-fullfilling prophecy.", 
        "Keeper Tax is a YC-backed, four person team that has a multi-year runway, $1.5M ARR, and a massive market We build software to automate bookkeeping and tax filing for freelancers. There are 60M people in the US who receive 1099 income (dog walkers, handymen, online sellers). Our software connects to users' bank accounts, ingests bank statements, and automatically finds write-offs. On average, users find $6,000 in tax deductions per year using our software.t You will be the fifth team member, working alongside a passionate, mission-driven team to help busy freelancers save time and money. We've raised a $13M Series A from top investors, including Y Combinator, Matrix, Foundation, and e.ventures, so we have ample runway to execute on our mission.", 
        "Ethereum will be completing the transition to Proof-of-Stake in the upcoming months, which brings a myriad of improvements that have been theorized for years. But now that the Beacon chain has been running for a few months, we can actually dig into the numbers. One area that we’re excited to explore involves new energy-use estimates, as we end the process of expending a country’s worth of energy on consensThere aren’t any concrete statistics on energy consumption (or even what hardware is used) as of yet, so what follows is a ball-park estimation of the energy consumption of the future of EthereumAs many people are running multiple validators, I’ve decided to use the number of unique addresses that made deposits as a proxy for how many servers are out there today. Many stakers could have used multiple eth1 addresses, but this largely cancels out against those with redundant setupsAt the time of writing, there are 140,592 validators from 16,405 unique addresses. Obviously this is heavily skewed by exchanges and staking services, so removing them leaves 87,897 validators assumed to be staking from home. As a sanity check, this implies that the average home-staker runs 5.4 validators which seems like a reasonable estimate to me.",
        "We've always had a soft spot for language at Google. Early on, we set out to translate the web. More recently, we’ve invented machine learning techniques that help us better grasp the intent of Search queries. Over time, our advances in these and other areas have made it easier and easier to organize and access the heaps of information conveyed by the written and spoken wordBut there’s always room for improvement. Language is remarkably nuanced and adaptable. It can be literal or figurative, flowery or plain, inventive or informational. That versatility makes language one of humanity’s greatest tools — and one of computer science’s most difficult puzzles.LaMDA, our latest research breakthrough, adds pieces to one of the most tantalizing sections of that puzzle: conversation",
        "Running queries on Uber’s data platform lets us make data-driven decisions at every level, from forecasting rider demand during high traffic events to identifying and addressing bottlenecks in the driver sign-up process. Our Apache Hadoop-based data platform ingests hundreds of petabytes of analytical data with minimum latency and stores it in a data lake built on top of the Hadoop Distributed File System (HDFS)Our data platform leverages several open source projects (Apache Hive, Apache Presto, and Apache Spark) for both interactive and long running queries, serving the myriad needs of different teams at Uber. All of these services were built in Java or Scala and run on open source Java Virtual Machine (JVM).  Uber’s growth over the last few years exponentially increased both the volume of data and the associated access loads required to process it, resulting in much more memory consumption from services. Increased memory consumption exposed a variety of issues, including long garbage collection (GC) pauses, memory corruption, out-of-memory (OOM) exceptions, and memory leaks. ",
        "They describe an increasingly clear correlation between poor mental health outcomes and social media use, and they worry that Facebook (which also owns Instagram and WhatsApp) in particular may be muddying the waters on that connection to protect its public image.'The correlational evidence showing that there is a link between social media use and depression is pretty definitive at this point,'said Jean Twenge, a psychology professor at San Diego State University. 'The largest and most well-conducted studies that we have all show that teens who spend more time on social media are more likely to be depressed or unhappy.Correlation is not causation, and one area of further study is whether greater social media usage leads to poor mental health outcomes or whether those who are depressed and unhappy are drawn to spend more time on social media. But researchers also worry that not enough government funding is going toward getting objective data to answer these sorts of questionsFacebook also almost certainly knows more than it has publicly revealed about how its products affect people.",
        "People love being together — to share, collaborate and connect.  And this past year, with limited travel and increased remote work, being together has never felt more importantThrough the years, we’ve built products to help people feel more connected. We’ve simplified email with Gmail, and made it easier to share what matters with Google Photos and be more productive with Google Meet. But while there have been advances in these and other communications tools over the years, they're all a far cry from actually sitting down and talking face to faceWe looked at this as an important and unsolved problem. We asked ourselves: could we use technology to create the feeling of being together with someone, just like they're actually thereTo solve this challenge, we’ve been working for a few years on Project Starline — a technology project that combines advances in hardware and software to enable friends, families and coworkers to feel together, even when they're cities (or countries) apartImagine looking through a sort of magic window, and through that window, you see another person, life-size and in three dimensions. You can talk naturally, gesture and make eye contact",
        "Use your existing security key for Git operations. When used for SSH operations, security keys move the sensitive part of your SSH key from your computer to a secure external security key. SSH keys that are bound to security keys protect you from accidental private key exposure and malware. You perform a gesture, such as a tap on the security key, to indicate when you intend to use the security key to authenticate. This action provides the notion of “user presence.”. Security keys are not limited to a single application, so the same individual security key is available for both web and SSH authentication. You don’t need to acquire a separate security key for each use case. And unlike web authentication, two-factor authentication is not a requirement when using security keys to authenticate to Git. As always, we recommend using a strong password, enrolling in two-factor authentication, and setting up account recovery mechanisms. Conveniently, security keys themselves happen to be a great recovery option for securely retaining access to your two-factor-enabled account if you lose access to your phone and backup codes.",
        "NVIDIA announced today that it's halving the hash rate for Etehereum cryptocurrency mining on the new GeForce RTX 3080, 3070, and 3060 Ti graphics cards to make them less desirable for miners.The company will add 'Lite Hash Rate' or 'LHR' identifiers to retail product listings and boxes for all these new nerfed graphics cards that will start shipping later this month'Today, we're taking additional measures by applying a reduced ETH hash rate to newly manufactured GeForce RTX 3080, RTX 3070 and RTX 3060 Ti graphics cards,' said Matt Wuebbling, NVIDIA's Global Head of GeForce Marketing'This reduced hash rate only applies to newly manufactured cards with the LHR identifier and not to cards already purchased.According to Wuebbling, this decision was taken to make sure that more of these cards will be used by gamers worldwide instead of stacked in cryptocurrency mining farms.",
        "Beginner’s Luck. When author and Outside contributing editor Tom Vanderbilt had his daughter, he, like so many other new parents, spent endless hours in awe of her capacity to learn new things and the joy those processes brought her. This got Vanderbilt thinking: When was the last time I learned anything new? So began his journey to learn five new skills—chess, singing, surfing, drawing, and juggling—which he details in his latest book, Beginners: The Joy and Transformative Power of Lifelong Learning. Vanderbilt makes a compelling case that learning something new has myriad advantages, including promoting the brain’s ability to rewire itself, connecting you to new people and new communities, and reengaging our innate curiosity and open-mindedness. While all of these offer tremendous benefits, that last one may be the most important.",
        "After a better part of a year of work, I am excited to show you a brand new Map of the Internet, up to date for the year 2021.Inspired by design of historical maps, this project aims to concisely, but still comprehensively visualize the current state of the World Wide Web, and document the largest and most popular websites over the period of 2020-2021, along with their countless aspects and featuresCompared to any previous iteration of the Map of the Internet, this new version is many times more detailed and informative. It includes several thousand of some of the most popular websites, represented as distinct 'countries', which are grouped together with others of similar type or category, forming dozens of distinct clusters, regions and continents that stretch throughout the map, such as 'news sites', 'search engines', 'social networks', 'e-commerce', 'adult entertainment', 'file sharing', 'software companies' and so much more.  In the center of it all can be found ISPs and web browsers, which form the core and backbone of the internet as we know it, while the far south is the domain of the mysterious 'dark web'Color schemes of websites are based on the dominant colors of their user interface or logo. To add further detail and provide deeper insight, many features and services provided by these websites, their sections and content categories, as well as distinct content creators, are labeled as cities and towns (which number at well over 10 thousand). Website founders and CEOs are represented as capital cities, while hundreds of the most popular users of social networks and celebrities can be found in the realms of Youtube, Facebook, or Twitter. Mountains, hills, seas and valleys represent a wide variety of aspects of the internet, its culture and computer science overall, while almost a hundred of some of the most important internet and computing pioneers are also featured on the map in the names of underwater ridges. "
]
```

<p>Запустим наш код снова и увидим: </p>

```
[1.0, 0.9951382821672096, 0.013216669455351429, 0.006211233733302461, 0.009507461374260949, 0.0, 0.020128203442007216, 0.008342148671695554, 0.0073409736036723, 0.015846664913869494, 0.028675059137806467, 0.01902343254541515, 0.012055207195046354, 0.0, 0.0033563157130868005, 0.018957545252774887, 0.014255796878704553]
```

<p>Второе значение (наш добавленный и очень похожий на первое описание текст) явно меньше единицы и можно заявить, что он отличается, от первого текста, но он же и очень похож на первый текст (т.е. косинусное расстояние близится к единице), так как разница в два слова совсем невелика.</p>
<br>
<p>Мы с вами разобрали два основных подхода к построению рекомендательных систем. Оба из них содержат свои плюсы и минусы. Самыми идеальными вариантами для сайта или приложения очень часто служат гибридные РС, которые содержат в себе те или иные фишки этих двух систем. Во всяком случае, за основу почти всегда берется один из выше перечисленных вариантов рекомендательных систем.</p>
<br>
<h3>Спасибо за внимание!</h3>
