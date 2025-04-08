import random


def generate_stimuli_list():
    stimuli_labels = ['A', 'B', 'C', 'D', 'E', 'F']  # Six stimuli
    system_labels = ['A', 'B', 'C', 'D']  # Four systems

    stimuli_list = [f"{i + 1}{stimuli}{system}" for i, stimuli in enumerate(stimuli_labels) for system in system_labels]
    return stimuli_list


def randomize_stimuli(stimuli_list):
    grouped_stimuli = {stimulus: [] for stimulus in 'ABCDEF'}

    for item in stimuli_list:
        stimulus = item[1]
        grouped_stimuli[stimulus].append(item)

    grouped_list = list(grouped_stimuli.values())
    random.shuffle(grouped_list)

    for group in grouped_list:
        random.shuffle(group)

    randomized_list = [item for group in grouped_list for item in group]

    return randomized_list


if __name__ == "__main__":
    base_list = generate_stimuli_list()
    randomized_list = randomize_stimuli(base_list)

    print(base_list)
    print(randomized_list)
