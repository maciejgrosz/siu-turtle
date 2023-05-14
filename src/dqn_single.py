# encoding: utf8
import random
from collections import deque

import tensorflow as tf
import numpy as np
from keras.layers import Conv3D, Permute, Dense, Flatten
from keras.models import Sequential
from tensorflow import keras

from turtlesim_env_base import TurtlesimEnvBase


class DqnSingle(object):
    """Inicjalizacja parametrami domyślnymi, przechowanie dostarczonej referencji na środowisko symulacyjne"""

    def __init__(self, env: TurtlesimEnvBase, id_prefix='dqns', seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.id_prefix = id_prefix  # przyrostek identyfikatora modelu

        self.DISCOUNT = .8                      # D dyskonto dla nagrody w następnym kroku   #z jaką dokładnością ufamy tej decyzji w następnym kroku
        self.EPS_INIT = 1.0                     # *  ε początkowy       -nie ruszać
        self.EPS_DECAY = .99                    # *E spadek ε           -nie ruszać
        self.EPS_MIN = .05                      # *e ε minimalny        -nie ruszać        #mówi w ilu procentach wykonamy ruch losowy
        self.REPLAY_MEM_SIZE_MAX = 20_000       # M rozmiar cache decyzji
        self.REPLAY_MEM_SIZE_MIN = 1_500        # m zapełnienie warunkujące uczenie (4_000)
        self.MINIBATCH_SIZE = 32                # B liczba decyzji w próbce uczącej
        self.TRAINING_BATCH_SIZE = self.MINIBATCH_SIZE // 4
        self.UPDATE_TARGET_EVERY = 20           # U co ile treningów aktualizować model wolnozmienny
        self.EPISODES_MAX = 4000                # *P liczba epizodów uczących, ile razy będzeimy trenować sieć zanim zacznie z niej korzystać   -nie ruszać
        self.CTL_DIM = 10                       # liczba możliwych akcji (tj. sterowań, decyzji) są to możliwe ruchy         -    niedawać jako prędkość w scenariuszu 0 bo może się zaciąć
        self.TRAIN_EVERY = 4                    # T co ile kroków uczenie modelu szybkozmiennego
        self.SAVE_MODEL_EVERY = 200             # *  co ile epizodów zapisywać model # TODO-STUDENCI

        self.model = None
        self.target_model = None
        self.replay_memory = None

    def xid(self) -> str:
        """Sygnatura eksperymentu, tj. wartości parametrów w jednym łańcych znaków.
        Używane do nazywania plików z wynikami.
            2 litery - parametr środowiska
            1 litera - parametr klasy uczącej

        :return: str
        """
        return f'{self.id_prefix}-Gr{self.env.GRID_RES}_Cr{self.env.CAM_RES}_Sw{self.env.SPEED_RWRD_RATE}' \
               f'_Sv{self.env.SPEED_RVRS_RATE}_Sf{self.env.SPEED_FINE_RATE}_Dr{self.env.DIST_RWRD_RATE}' \
               f'_Oo{self.env.OUT_OF_TRACK_FINE}_Cd{self.env.COLLISION_DIST}_Ms{self.env.MAX_STEPS}' \
               f'_Pb{self.env.PI_BY}_D{self.DISCOUNT}_E{self.EPS_DECAY}_e{self.EPS_MIN}_M{self.REPLAY_MEM_SIZE_MAX}' \
               f'_m{self.REPLAY_MEM_SIZE_MIN}_B{self.MINIBATCH_SIZE}_U{self.UPDATE_TARGET_EVERY}' \
               f'_P{self.EPISODES_MAX}_T{self.TRAIN_EVERY}'

    @staticmethod
    def ctl2act(decision: int):
        """Zakodowanie wybranego sterowania (0-5).
        Na potrzeby środowiska: (prędkość, skręt)
        prędkość/skręt -.1rad 0 .1rad

        :param decision: liczba całkowita od 0 do 5
        """
        v = .4 if decision >= 3 else .2
        w = .25 * (decision % 3 - 1)
        return [v, w]

    @staticmethod
    def inp_stack(last, cur):
        """Złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x8 wejścia do sieci
        fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)

        :param last: last?
        :param cur: current?
        :return:
        """
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], last[2], last[3], last[4], last[5]], axis=-1)
        return inp

    def decision(self, the_model, last, cur):
        """Predykcja nagród łącznych (Q) za sterowania na podst. bieżącej i ostatniej sytuacji

        :param the_model:
        :param last:
        :param cur:
        :return:
        """
        inp = np.expand_dims(self.inp_stack(last, cur), axis=-1)
        inp = np.expand_dims(inp, axis=0)
        return the_model(inp).numpy().flatten()  # wektor przewidywanych nagród dla sterowań
        # UBYTEK PAMIĘCI w dockerze:
        # return the_model.predict(inp,verbose=0).flatten()

    def make_model(self):
        """Wytworzenie modelu - sieci neuronowej

        :return:
        """
        n = self.env.GRID_RES  # rozdzielczość rastra
        m = 8  # liczba warstw z inp_stack()

        self.model = Sequential()
        self.model.add(Conv3D(filters=2 * m, kernel_size=(2, 2, m), activation='relu', input_shape=(n, n, m, 1)))
        self.model.add(Permute((1, 2, 4, 3)))
        self.model.add(Conv3D(filters=2 * m, kernel_size=(2, 2, 2 * m), activation='relu'))
        self.model.add(Permute((1, 2, 4, 3)))
        self.model.add(Conv3D(filters=2 * m, kernel_size=(2, 2, 2 * m), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.CTL_DIM, activation="linear"))  # wyjście Q dla każdej z CTL_DIM decyzji
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

    def train_main(self, tname: str, save_model=True):
        """Uczenie od podstaw:
            - generuj kroki
            - gromadź pomiary
            - ucz na próbce losowej
            - okresowo aktualizuj model pomocniczy

        :param tname:
        :param save_model:
        :return:
        """
        self.target_model = keras.models.clone_model(self.model)  # model pomocniczy (wolnozmienny)
        self.replay_memory = deque(maxlen=self.REPLAY_MEM_SIZE_MAX)  # historia kroków

        episode_rewards = np.zeros(self.EPISODES_MAX) * np.nan  # historia nagród w epizodach
        epsilon = self.EPS_INIT
        step_cnt = 0
        train_cnt = 0

        for episode in range(self.EPISODES_MAX):  # ucz w epizodach treningowych
            print(f'{len(self.replay_memory)} E{episode} ', end='')
            current_state = self.env.reset(tnames=[tname], sections=['random'])[tname].map
            last_state = [i.copy() for i in current_state]  # zaczyna od postoju: poprz. stan taki jak obecny
            episode_rwrd = 0  # suma nagród za kroki w epizodzie

            while True:  # o przerwaniu decyduje do_train()
                if np.random.random() > epsilon:  # sterowanie wg reguły albo losowe
                    control = np.argmax(self.decision(self.model, last_state, current_state))
                    print('o', end='')  # "o" - sterowanie z modelu
                else:
                    control = np.random.randint(0, self.CTL_DIM)  # losowa prędkość pocz. i skręt
                    print('.', end='')  # "." - sterowanie losowe

                new_state, reward, done = self.env.step({tname: self.ctl2act(control)})  # krok symulacji
                step_cnt += 1
                episode_rwrd += reward
                self.replay_memory.append((last_state, current_state, control, reward, new_state, done))

                # bufor ruchów dość duży oraz przyszła pora by podtrenować model
                if len(self.replay_memory) >= self.REPLAY_MEM_SIZE_MIN and step_cnt % self.TRAIN_EVERY == 0:
                    self.do_train()  # ucz, gdy zgromadzono dość próbek
                    train_cnt += 1

                    if train_cnt % self.UPDATE_TARGET_EVERY == 0:
                        self.target_model.set_weights(self.model.get_weights())  # aktualizuj model pomocniczy
                        print('T', end='')
                    else:
                        print('t', end='')

                if done:
                    break

                last_state = current_state  # przejście do nowego stanu
                current_state = new_state  # z zapamiętaniem poprzedniego

                if epsilon > self.EPS_MIN:  # rosnące p-stwo uczenia na podst. historii
                    epsilon *= self.EPS_DECAY
                    epsilon = max(self.EPS_MIN, epsilon)  # podtrzymaj losowość ruchów

            episode_rewards[episode] = episode_rwrd
            print(f' {np.nanmean(episode_rewards[episode - 19:episode + 1]) / self.env.MAX_STEPS:.2f}')  # śr. nagroda za krok

            # TODO-STUDENCI - Okresowy zapis modelu
            if save_model is True and episode > 0 and (episode + 1) % self.SAVE_MODEL_EVERY == 0:
                self.model.save(f'siu/models/{self.xid()}.h5')

    def do_train(self, episode=None):
        """Przygotowuje próbkę uczącą i wywołuje douczanie modelu"""
        print(episode)
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)  # losowy podzbiór kroków z historii
        q_zero = np.zeros((self.MINIBATCH_SIZE, self.CTL_DIM))  # nagrody krok n wg modelu bieżącego
        q1_target = q_zero.copy()  # nagrody krok n+1 wg modelu pomocniczego

        for idx, (last_state, current_state, _, _, new_state, _) in enumerate(minibatch):
            q_zero[idx] = self.decision(self.model, last_state, current_state)  # krok n / model bieżący
            q1_target[idx] = self.decision(self.target_model, current_state, new_state)  # krok n+1 / model główny

        x = []  # sytuacje treningowe
        y = []  # decyzje treningowe

        for idx, (last_state, current_state, control, reward, new_state, done) in enumerate(minibatch):
            if done:
                new_q = reward  # nie ma już stanu następnego, ucz się otrzymywać faktyczną nagrodę
            else:
                # nagroda uwzględnia nagrodę za kolejny etap sterowania
                new_q = reward + self.DISCOUNT * np.max(q1_target[idx])

            q0 = q_zero[idx].copy()
            q0[control] = new_q  # pożądane wyjście wg informacji po kroku (reszta - oszacowanie)
            inp = self.inp_stack(last_state, current_state)  # na wejściu - stan

            x.append(np.expand_dims(inp, axis=-1))
            y.append(q0)

        # douczanie modelu (w niektórych implementacjach wywoływana bezpośrednio propagacja wsteczna gradientu)
        # zapamiętanie wag tylko 1. warstwy splotowej
        x = np.stack(x)
        y = np.stack(y)
        self.model.fit(x, y, batch_size=self.TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        
        # TODO-STUDENCI
        weights = np.copy(self.model.weights[0])
        weights[:, :, 8:, :, :] = np.copy(self.model.weights[0][:, :, 8:, :, :])
        self.model.weights[0] = tf.Variable(weights)
