import csv

DATASET_NAME = 'FSD50k'


class ProcessFSD50k:

    @staticmethod
    def extract_mappings(datasets_path, mappings):
        dataset_path = datasets_path + DATASET_NAME + '/'
        mappings = ProcessFSD50k.extract_custom(dataset_path, 'dev', mappings)
        mappings = ProcessFSD50k.extract_custom(dataset_path, 'eval', mappings)

        return mappings

    @staticmethod
    def extract_custom(dataset_path, type, mappings):
        mappings_file = dataset_path + type + '.csv'
        audio_folder = dataset_path + 'audio-' + type
        with open(mappings_file, newline='') as f:
            reader = csv.reader(f)
            iterreader = iter(reader)
            next(iterreader)
            for row in iterreader:
                categories = row[1].split(',')
                for category in categories:
                    for class_maps in mappings:
                        for dataset in mappings[class_maps]['datasets']:
                            if category == mappings[class_maps]['datasets'][dataset]['matching_category']:
                                mappings[class_maps]['count'] += 1
                                mappings[class_maps]['mappings'].append(
                                    [audio_folder + "/" + row[0] + ".wav", str(0), mappings[class_maps]['id'],
                                     category])
        return mappings

    @staticmethod
    def filter(datasets_path, mappings):
        filters_file = datasets_path + DATASET_NAME + '/exclude_files_filter.csv'

        unwanted_files = []
        with open(filters_file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if DATASET_NAME == row[1]:
                    unwanted_files.append(row[0])

        for category in mappings:
            maps = mappings[category]['mappings']
            new_maps = []
            for mapping in maps:
                if mapping[0].split('/')[-1] not in unwanted_files:
                    new_maps.append(mapping)

            mappings[category]['mappings'] = new_maps
            mappings[category]['count'] = len(new_maps)

        return mappings
