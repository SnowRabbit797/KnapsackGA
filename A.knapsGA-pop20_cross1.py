import random
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

# 各品物の価値（20個）
value  = [60, 100, 120, 80, 90, 40, 70, 50, 200, 150, 180, 75, 110, 95, 130, 55, 105, 125, 85, 65]

# 各品物の重さ（20個）
weight = [10, 20, 30, 15, 25, 5, 18, 12, 35, 28, 33, 16, 22, 19, 24, 14, 17, 21, 13, 11]

# ナップサックの容量
capacity = 150

# 品物の数
n = len(value)

# 初期個体群のサイズ（個体数）
popSize = 50

# 突然変異率
mutation_rate = 0.1

# 世代数
generations = 10000

# 初期個体群の生成
# 各個体はn個の0または1のリストで表される
initialPopulation = [[random.randint(0,1) for i in range(n)] for i in range(popSize)]

# 適応度関数
# 各個体の価値の総和を計算し、容量を超えた場合は0を返す
def fitness(individual):
    total_value = np.dot(individual, value) #np.dotはnumpyの内積計算で使われる
    total_weight = np.dot(individual, weight)
    if total_weight > capacity:
        return 0 # 容量を超えた場合は無効な解
    return total_value

# 選択関数
# ルーレット選択法を用いて次世代の個体を選ぶ
def selection(population):
    weighted_population = [(individual, fitness(individual)) for individual in population] #個体とその適応度のペアのリストを作成
    weighted_population = [individual for individual in weighted_population if individual[1] > 0] #weighted_populationから適応度が0以下の個体を除外
    if not weighted_population: #すべての個体が適応度0の場合に一世代前の個体を返す
        return population
    total_fitness = sum(f for _, f in weighted_population) #ルーレット選択の分母(適応度の総和)を計算
    selection_probs = [f / total_fitness for _, f in weighted_population] #選択確率をリストとして表している。"_"は無意味を表す記号 
    selected_indices = np.random.choice(len(weighted_population), size=popSize, replace=True, p=selection_probs) #np.random.choiceは与えられた確率にそって値を抽出するもの。Trueは重複を許している(同じ個体が選ばれるのを許可している)
    return [weighted_population[i][0] for i in selected_indices] #適応度と個体のペアのリストから個体だけを取り出す

# 一点交叉関数
# 親個体の部分を組み合わせて新しい個体を生成する
def crossover(parent1, parent2):
    point = random.randint(1, n - 1)
    return parent1[:point] + parent2[point:]

# 突然変異関数
# 確率的に各遺伝子を突然変異させる
def mutate(individual):
    return [1 - gene if random.random() < mutation_rate else gene for gene in individual] #もし突然変異がされた場合は、1-"リスト"をしている。(リストの値を反転させている。)

# 遺伝的アルゴリズムの実行
def genetic_algorithm():
    population = initialPopulation #①初期個体群の生成
    best_individual = max(population, key=fitness) #この後のエリート戦略で使用するために適応度が最大のものを抜き出す
    best_value = fitness(best_individual)
    best_values = [best_value] # 最適な適応度を保存するリスト
    
    for generation in range(generations):
        population = selection(population) #②適応度の評価と、それに応じて③選択をする
        next_generation = []
        
        #エリート戦略：最良の個体を次世代にそのまま残す
        next_generation.append(best_individual) #④エリート戦略
        
        for i in range(0, popSize - 1, 2): #親を2体ずつ選ぶため、rangeの2ずつでfor文を回している
            parent1 = population[i]
            parent2 = population[(i+1) % popSize]
            child1 = crossover(parent1, parent2) #⑤交叉
            child2 = crossover(parent2, parent1)
            next_generation.append(mutate(child1))#⑥突然変異
            next_generation.append(mutate(child2))
        
        population = next_generation[:popSize] #遺伝オペレータにより、次世代の個体数が制限個体数を上回っている可能性がある。突然変異が起きた場合、ただappendしているだけなので、個体数がpopSizeより大きくなってしまう。なので、ここでスライス(popSizeの大きさに戻す)している。
        
        
        #各世代の最適な個体とその適応度を表示(ここも関係ない)
        current_best_individual = max(population, key=fitness)
        current_best_value = fitness(current_best_individual)
        
        if current_best_value > best_value: #best_valueの更新
            best_individual = current_best_individual
            best_value = current_best_value
        
        best_values.append(best_value) #各世代の最適な適応度を保存
        
        print(f"Generation {generation + 1}: Best Value = {best_value}")
        
    return best_individual, best_value, best_values

#遺伝的アルゴリズムを実行し、最適解を取得
best_solution, best_value, best_values = genetic_algorithm()
print("\nFinal Best Solution:")
pprint(best_solution)
print("Final Best Value:", best_value)

#適応度の推移をプロット
plt.plot(best_values)
plt.xlabel('Generation')
plt.ylabel('Best Value')
plt.title('Best Value over Generations')
plt.show()
