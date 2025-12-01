import random
import itertools
import matplotlib.pyplot as plt

products = [
    ("Апельсин", 47, 0.9, 0.1, 11.8, 12),
    ("Груша", 57, 0.4, 0.1, 15, 14),
    ("Овсянка", 366, 12.0, 6.2, 68, 15),
    ("Картофель варёный", 82, 2.0, 0.3, 19, 5),
    ("Куриная грудка", 165, 31.0, 3.6, 0, 32),
    ("Йогурт натуральный", 61, 3.5, 3.3, 4.7, 25),
    ("Сыр твёрдый", 330, 25, 27, 0, 70),
    ("Сыр плавленый", 280, 18, 22, 2.5, 50),
    ("Капуста", 27, 1.8, 0.1, 5.4, 6),
    ("Форель", 208, 20, 13, 0, 90),
    ("Макароны", 131, 5, 1.1, 25, 10),
    ("Огурцы", 15, 0.8, 0.1, 3.6, 8),
    ("Сок яблочный", 46, 0.1, 0.1, 11, 6),
    ("Печенье", 440, 7.0, 14, 71, 65),
    ("Шоколад", 546, 6.0, 35, 52, 80),
    ("Пицца замороженная", 280, 11, 12, 30, 110),
    ("Сосиски", 240, 12, 20, 2, 85),
]


num_products = len(products)
k = 9
max_price = 300
target_calories = 2000
target_proteins = 160
target_carbs = 200
target_fats = 70


def fitness(chromosome):
    total_price = sum(products[i][5] for i in chromosome)
    total_calories = sum(products[i][1] for i in chromosome)
    total_proteins = sum(products[i][2] for i in chromosome)
    total_fats = sum(products[i][3] for i in chromosome)
    total_carbs = sum(products[i][4] for i in chromosome)

    if total_price > max_price:
        return 0

    calories_diff = abs(total_calories - target_calories)
    proteins_diff = abs(total_proteins - target_proteins)
    carbs_diff = abs(total_carbs - target_carbs)
    fats_diff = abs(total_fats - target_fats)

    return 1 / (calories_diff + proteins_diff + fats_diff + carbs_diff + 1e-6)  # добавлен +1e-6 для избежания деления на 0


def brute_force_search():
    best_solution = None
    best_fitness = -1

    for combo in itertools.combinations(range(num_products), k):
        f = fitness(combo)
        if f > best_fitness:
            best_fitness = f
            best_solution = combo

    return best_solution, best_fitness


def generate_chromosome():
    return random.sample(range(num_products), k)


def crossover_one_point(parent1, parent2):
    point = random.randint(1, k - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def crossover_two_point(parent1, parent2):
    point1, point2 = sorted([random.randint(1, k - 1), random.randint(1, k - 1)])
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


def crossover_uniform(parent1, parent2):
    child1 = [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]
    child2 = [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]
    return child1, child2


def mutate_random(chromosome):
    idx = random.randint(0, k - 1)
    new_gene = random.randint(0, num_products - 1)
    chromosome[idx] = new_gene
    return chromosome


def mutate_swap(chromosome):
    idx1, idx2 = random.sample(range(k), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


def mutate_inversion(chromosome):
    idx1, idx2 = sorted(random.sample(range(k), 2))
    chromosome[idx1:idx2 + 1] = reversed(chromosome[idx1:idx2 + 1])
    return chromosome


def genetic_algorithm(population_size, generations, crossover_method, mutation_method):
    population = [generate_chromosome() for _ in range(population_size)]
    best_solution = None
    best_fitness = -1
    fitness_history = []

    for gen in range(generations):
        population_fitness = [(chromosome, fitness(chromosome)) for chromosome in population]
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        best_chromosome, best_f = population_fitness[0]

        if best_f > best_fitness:
            best_fitness = best_f
            best_solution = best_chromosome

        selected_parents = population_fitness[:population_size // 2]

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(selected_parents)[0], random.choice(selected_parents)[0]
            child1, child2 = crossover_method(parent1, parent2)
            new_population.append(mutation_method(child1))
            new_population.append(mutation_method(child2))

        population = new_population
        fitness_history.append(best_fitness)

    return best_solution, fitness_history


def main():
    population_size = 10
    generations = 100


    print("\nПолный перебор")
    bf_solution, bf_fitness = brute_force_search()
    print(f"Лучший рацион (полный перебор): {', '.join([products[i][0] for i in bf_solution])}")
    print(f"Цена: {sum(products[i][5] for i in bf_solution)}")
    print(f"Калорийность: {sum(products[i][1] for i in bf_solution)}")
    print(f"Белки: {sum(products[i][2] for i in bf_solution)}")
    print(f"Жиры: {sum(products[i][3] for i in bf_solution)}")
    print(f"Углеводы: {sum(products[i][4] for i in bf_solution)}")

    crossover_methods = [
        ("Одноточечное скрещивание", crossover_one_point),
        ("Двуточечное скрещивание", crossover_two_point),
        ("Униформное скрещивание", crossover_uniform)
    ]

    mutation_methods = [
        ("Случайная мутация", mutate_random),
        ("Мутация обменом", mutate_swap),
        ("Мутация инверсией", mutate_inversion)
    ]

    plt.figure(figsize=(15, 15))

    for i, (crossover_name, crossover_method) in enumerate(crossover_methods):
        for j, (mutation_name, mutation_method) in enumerate(mutation_methods):
            best_solution, fitness_history = genetic_algorithm(population_size, generations, crossover_method,
                                                               mutation_method)

            plt.subplot(3, 3, i * 3 + j + 1)
            plt.plot(fitness_history)
            plt.xlabel('Поколение')
            plt.ylabel('Приспособленность')
            plt.title(f'{crossover_name} + {mutation_name}')

            print(f"\n{crossover_name} + {mutation_name}:\n")
            print(f"Лучший рацион: {', '.join([products[i][0] for i in best_solution])}")
            print(f"Цена: {sum(products[i][5] for i in best_solution)}")
            print(f"Калорийность: {sum(products[i][1] for i in best_solution)}")
            print(f"Белки: {sum(products[i][2] for i in best_solution)}")
            print(f"Жиры: {sum(products[i][3] for i in best_solution)}")
            print(f"Углеводы: {sum(products[i][4] for i in best_solution)}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
