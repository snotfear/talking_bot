import torch
import numpy as np
from random import random, randrange
import telebot


from sklearn.metrics.pairwise import cosine_similarity

from numpy.linalg import norm
from navec import Navec
from slovnet.model.emb import NavecEmbedding
from razdel import tokenize


'''Загрузим готовый эмбендинг Navec от проекта Natasha'''
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

'''Загрузим файл с вопрос-ответами'''
question_dict = dict()
temp_list = []
'''Токен и запуск бота'''
token = '5069703088:AAEA-gfnn-tn9yy6i616eNtiV4sEUJa3x4E'
bot = telebot.TeleBot(token)

with open('question-answer.txt', 'r', encoding='utf-8') as file:
    temp_data = file.read().splitlines()
    '''И разобъём строки по табуляции (ключи) и точке-с-запятой (значения)
    Ключами будут известные вопросы, значения - варианты ответов на них {'вопрос':[ответ_1, ответ_2...]}
    '''
    for line in temp_data:
        temp_list.append(line.split('\t'))
        for keys_and_value in temp_list:
            question_dict[keys_and_value[0]] = keys_and_value[1].split(';')
        temp_list = []

def generate_list_of_question(dict_with_q_a):
    '''
    :param dict_with_q_a: Словарь с вопросами и ответами
    :return: Список вопросов, будет нужен для функции ранжирования и возврата нужного ответа
    '''
    return [i for i in dict_with_q_a.keys()]


'''для разбивки на токены применяется функция tokenize, обернём её в класс для удобства'''
class MyTokenizer:
    def __init__(self):
        pass
    def tokenizes(self, text):
        sub_token = tokenize(text)
        return [_.text for _ in sub_token]

'''и сразу создадим объект класса, который будем использовать во всех функциях'''
tokenizer = MyTokenizer()


def question_to_vec(question, embeddings, tokenizer, dim=300):
    '''
    Функция перевода предложения в вектор
    :param question: строка предложения
    :param embeddings: эмбеддинг (navec)
    :param tokenizer: токенизатор (razdel)
    :return: векторное представления всего вопроса
    '''
    question = question.lower()
    summ = 0
    count = 0
    tokkens = tokenizer.tokenizes(question)

    for i in tokkens:
        if i in embeddings:
            summ += embeddings[i]
            count += 1
    if count == 0:
        return np.zeros(dim)
    return summ/count


def rank_candidates(question, candidates_vec, embeddings, tokenizer, dim=300):
    '''
    Функция ранжирования из списка вопросов
    :param question: строка вопроса
    :param candidates_vec: список известных вопросов в виде векторных представлений
    :param embeddings: эмбеддинг
    :param tokenizer: токенизатор
    :return: вернёт индекс наиболее близкого вопроса из базы вопросов
    '''

    data = []
    q_vec = question_to_vec(question, embeddings, tokenizer, dim)
    for ind, candidate_vec in enumerate(candidates_vec):
        cos_dist = 1 - cos_simm(q_vec, candidate_vec)
        data.append([cos_dist, ind, candidate_vec])
    data.sort(key=lambda x: x[0], reverse=False)
    return data[0][1]


def cos_simm (v1, v2):
    '''
    :param v1 и v2: массивы - векторные представления слов или предложений
    :return: Косинусную разность, float
    '''
    return np.array(v1 @ v2.T /norm(v1) / norm(v2))

def random_for_answer(index_q):
    '''
    Функция для случайного выбора ответа из списка представленных
    :param index_q: индекс вопроса, который ближе всех по смыслу к заданному пользователем
    :return: случайный индекс для ответа
    '''
    random_int = randrange(0, len(question_dict[question_list[index_q]]))
    return random_int

'''Преобразуем известные вопросы в список с векторными представлениям для более быстрого сравнения'''

vectors_of_question = []
question_list = generate_list_of_question(question_dict)
for question in question_list:
    vectors_of_question.append(question_to_vec(question=question, embeddings=navec, tokenizer=tokenizer))


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Я бот, созданный для общения")


@bot.message_handler(content_types=['text'])
def send_text(message):
    user_q = str(message.text.lower())
    rank = rank_candidates(question=user_q, candidates_vec=vectors_of_question, embeddings=navec, tokenizer=tokenizer)
    answer = question_dict[question_list[rank]][random_for_answer(rank)]
    bot.send_message(message.chat.id, answer)


bot.infinity_polling()
'''
+Функция преобразования 
+ Реализовать сам словарь
+ Нужно добавить загрузку словаря с вопросами-ответами
фильтр матерных слов через отдельный список
+ сравнение косинусного расстояния 
Функцию добавления вопроса в словарь вопрос-ответ
индекс подходящего вопроса - сам вопрос - вариант ответа
рандомайзер для вариантов ответа'''
