import pandas as pd


def parse_random_orders(txt_path):
    random_dict = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_name = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.endswith(':'):
            current_name = line[:-1].strip()
            random_dict[current_name] = []

        else:
            codes_str = line.replace('[', '').replace(']', '').replace("'", "").replace(',', '')
            codes = codes_str.split()
            random_dict[current_name].extend(codes)

    return random_dict


def map_code_to_stim_sys(code):
    digit_to_stim = {
        '1': 'A',
        '2': 'B',
        '3': 'C',
        '4': 'D',
        '5': 'E',
        '6': 'F'
    }
    first_char = code[0]
    third_char = code[2]  # 'A', 'B', 'C', or 'D'

    stimulus = digit_to_stim.get(first_char, 'Unknown')
    system = third_char
    return stimulus, system


def clean_mushra_data(
        raw_qualtrics_path,
        random_orders_path,
        output_csv_path=None,
):
    df_raw = pd.read_csv(raw_qualtrics_path, header=0)

    cols_to_drop = list(range(0, 24)) + list(range(101, 106))
    df_cleaned = df_raw.drop(df_raw.columns[cols_to_drop], axis=1)

    all_notes_cols = [12, 25, 38, 51, 64]
    df_cleaned = df_cleaned.drop(df_cleaned.columns[all_notes_cols], axis=1)

    df_cleaned.columns = df_cleaned.iloc[0]
    df_cleaned = df_cleaned.drop(df_cleaned.index[0])
    df_cleaned = df_cleaned.drop(df_cleaned.index[0])
    df_cleaned.reset_index(drop=True, inplace=True)

    random_dict = parse_random_orders(random_orders_path)
    final_rows = []

    participant_names = list(random_dict.keys())

    for i in range(len(df_cleaned)):
        row = df_cleaned.iloc[i]

        if i >= len(participant_names):
            continue

        codes_list = random_dict[participant_names[i]]

        code_ratings_dict = {}

        for j, code in enumerate(codes_list):
            idx_start = 3 * j
            idx_end = idx_start + 3
            rating_vals = row[idx_start:idx_end].tolist()

            code_ratings_dict[code] = rating_vals

        sorted_codes = sorted(code_ratings_dict.keys())

        for code in sorted_codes:
            stim, sys = map_code_to_stim_sys(code)
            val1, val2, val3 = code_ratings_dict[code]

            final_rows.append({
                "Participant": participant_names[i],
                "Stimulus": stim,
                "System": sys,
                "Overall Envelopment and Immersion": val1,
                "Spatial & Temporal Quality": val2,
                "Spectral Quality": val3
            })

    df_final = pd.DataFrame(final_rows)

    if output_csv_path is not None:
        df_final.to_csv(output_csv_path, index=False)
        print(f"Cleaned MUSHRA data saved to {output_csv_path}")
    return df_final


if __name__ == "__main__":
    raw_data_path = "data/raw_data.csv"
    random_orders_path = "data/random_orders_list.txt"
    output_path = "data/cleaned_mushra_data.csv"

    df_clean = clean_mushra_data(
        raw_qualtrics_path=raw_data_path,
        random_orders_path=random_orders_path,
        output_csv_path=output_path
    )

    print(df_clean.head(20))
