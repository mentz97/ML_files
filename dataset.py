import json
import os
import collections
import typing
import torch
import transformers
import sklearn.preprocessing
import spacy
import tqdm

__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
trainFile = os.path.join(__location__, 'data/fashion_train_dials.json')
trainAPIFile = os.path.join(
    __location__, 'data/fashion_train_dials_api_calls.json')
valFile = os.path.join(__location__, 'data/fashion_dev_dials.json')
valAPIFile = os.path.join(
    __location__, 'data/fashion_dev_dials_api_calls.json')
testFile = os.path.join(__location__, 'data/fashion_devtest_dials.json')
testAPIFile = os.path.join(
    __location__, 'data/fashion_devtest_dials_api_calls.json')

le_actions = sklearn.preprocessing.LabelEncoder()
mlb_attributes = sklearn.preprocessing.MultiLabelBinarizer()

le_actions.fit(['SearchDatabase',
                'SearchMemory',
                'SpecifyInfo',
                'AddToCart',
                'None'])
mlb_attributes.fit([['ageRange',
                     'amountInStock',
                     'availableSizes',
                     'brand',
                     'clothingCategory',
                     'clothingStyle',
                     'color',
                     'customerRating',
                     'dressStyle',
                     'embellishment',
                     'forGender',
                     'forOccasion',
                     'hasPart',
                     'hemLength',
                     'hemStyle',
                     'info',
                     'jacketStyle',
                     'madeIn',
                     'material',
                     'necklineStyle',
                     'pattern',
                     'price',
                     'sequential',
                     'size',
                     'skirtLength',
                     'skirtStyle',
                     'sleeveLength',
                     'sleeveStyle',
                     'soldBy',
                     'sweaterStyle',
                     'waistStyle',
                     'warmthRating',
                     'waterResistance']])


class SentenceData():
    def __init__(self, turn_id: int, sentence: str, action: str, attributes: typing.List[str], dialogue_id: int) -> None:
        self.turn_id = turn_id
        self.sentence = sentence
        self.action = action
        self.attributes = attributes
        self.dialogue_id = dialogue_id


def __preprocess__(s: str) -> str:
    return s
    
class SIMMCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 train: bool = True,
                 test: bool = False,
                 concatenate: bool = False,
                 min_attribute_occ: int = 0,
                 exclude_attributes: typing.List[str] = [],
                 preprocess: typing.Callable[[str], str] = __preprocess__):
        api = self.__getapi__(train, test, concatenate=concatenate, min_attribute_occ=min_attribute_occ,
                              return_excluded_attributes=(min_attribute_occ > 0 or exclude_attributes), exclude_attributes=exclude_attributes, preprocess=preprocess)

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.encodings = self.tokenizer([d.sentence for d in api['results']],
                                        add_special_tokens=False,
                                        truncation=False,
                                        padding=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        self.attributes = mlb_attributes.transform(
            [d.attributes for d in api['results']])
        self.actions = le_actions.transform([d.action for d in api['results']])
        self.turn_ids = [d.turn_id for d in api['results']]
        self.dialogue_ids = [d.dialogue_id for d in api['results']]
        self.excluded_attributes = api['excluded_attributes'] if min_attribute_occ > 0 or exclude_attributes else [
        ]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item["attributes"] = torch.tensor(
            self.attributes[idx], dtype=torch.float32)
        item["action"] = torch.tensor(self.actions[idx])
        item["turn_id"] = self.turn_ids[idx]

        if self.dialogue_ids is not None:
            item["dialogue_id"] = self.dialogue_ids[idx]

        return item

    def __getapi__(self,
                   train: bool = False,
                   test: bool = False,
                   return_counter: bool = False,
                   return_excluded_attributes: bool = False,
                   concatenate: bool = False,
                   min_attribute_occ: int = 0,
                   exclude_attributes: typing.List[str] = [],
                   preprocess: typing.Callable[[str], str] = __preprocess__):
        actions = []
        attributes_list = []
        counter = []

        obj = {}

        with open(testFile if test else trainFile if train else valFile, 'r') as file:
            data = json.load(file)

            dialogues = list(
                map(lambda d: d['dialogue'], data['dialogue_data']))

            turn_ids = [sentence['turn_idx']
                        for dialogue in dialogues for sentence in dialogue]

            if concatenate:
                sentences = []
                for dialogue in tqdm.tqdm(dialogues):
                    concatenated = ""
                    for sentence in dialogue:
                        concatenated = "[CLS] " + preprocess(sentence['transcript']) + " [SEP] " if concatenated == "" else concatenated + \
                            preprocess(sentence['transcript']) + " [SEP] " 
                        sentences.append(concatenated)
            else:
                sentences = [preprocess(sentence['transcript'])
                             for dialogue in tqdm.tqdm(dialogues) for sentence in dialogue]

        with open(testAPIFile if test else trainAPIFile if train else valAPIFile, 'r') as file:
            data = json.load(file)

            dialogue_ids = [i['dialog_id'] for i in data for j in i['actions']]
            actions = [j['action'] for i in data for j in i['actions']]
            attributes_list = list(map(lambda x: x['attributes'] if x is not None else [], [
                j['action_supervision'] for i in data for j in i['actions']]))
            counter = collections.Counter(
                [y for x in list(filter(None, attributes_list)) for y in x])
            excluded_attributes = [key for key,
                                   val in counter.items() if val < min_attribute_occ or key in exclude_attributes]

            if return_counter:
                obj['counter'] = counter

            if return_excluded_attributes:
                obj['excluded_attributes'] = excluded_attributes

        obj['results'] = []
        for (turn_id, sentence, action, attributes, dialogue_id) in zip(turn_ids, sentences, actions, attributes_list, dialogue_ids):
            obj['results'].append(SentenceData(turn_id=turn_id,
                                               sentence=sentence,
                                               action=action,
                                               attributes=[
                                                   x for x in attributes if x not in excluded_attributes and x not in exclude_attributes],
                                               dialogue_id=dialogue_id))

        return obj


# def preprocess(sentence: str) -> str:
#     nlp = spacy.load("en_core_web_sm")
# 
#     doc = nlp(sentence)
#     for token in doc:
#         if token.pos_ == "VERB" or token.pos_ == "AUX":
#             sentence = sentence.replace(token.text, " " + token.lemma_, 1) if token.shape_.startswith(
#                 "'") else sentence.replace(token.text, token.lemma_, 1)
# 
#     return sentence
