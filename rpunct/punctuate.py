# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import logging
from langdetect import detect
from simpletransformers.ner import NERModel
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RPUNCT_LANG = os.environ.get("RPUNCT_LANG")
RPUNCT_USE_CUDA = os.environ.get("RPUNCT_USE_CUDA", "False")
print(f"RPUNCT_USE_CUDA: {RPUNCT_USE_CUDA}")
use_cuda_flag = RPUNCT_USE_CUDA.lower() == "true"

class RestorePuncts:
    def __init__(self, wrds_per_pred=250):
        self.wrds_per_pred = wrds_per_pred
        self.overlap_wrds = 30
        self.valid_labels = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']
        self.model_name = "felflare/bert-restore-punctuation"
        self.model = NERModel("bert", self.model_name, labels=self.valid_labels, use_cuda=use_cuda_flag, 
                              args={"silent": True, "max_seq_length": 512, "use_cuda": use_cuda_flag})
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def punctuate(self, text: str, lang:str=''):
        """
        Performs punctuation restoration on arbitrarily large text.
        Detects if input is not English, if non-English was detected terminates predictions.
        Overrride by supplying `lang='en'`
        
        Args:
            - text (str): Text to punctuate, can be few words to as large as you want.
            - lang (str): Explicit language of input text.
        """
        if not lang and len(text) > 10:
            #lang = detect(text)
            lang = RPUNCT_LANG
        if lang != 'en':
            raise Exception(F"""Non English text detected. Restore Punctuation works only for English.
            If you are certain the input is English, pass argument lang='en' to this function.
            Punctuate received: {text}""")

        # Step 1: Split the cleaned text into chunks within the 512-token limit
        # splits = self.split_on_toks(text, self.wrds_per_pred, self.overlap_wrds)
        splits = self.split_on_toks(text, max_length=512, overlap=30)
        
        # Step 2: Predict punctuation for each chunk
        # full_preds_lst = [self.predict(i['text']) for i in splits]
        full_preds_lst = [self.predict(chunk['text']) for chunk in splits]

        # Step 3: Extract predictions (discard logits)
        preds_lst = [i[0][0] for i in full_preds_lst]

        # join text slices
        combined_preds = self.combine_results(text, preds_lst, self.tokenizer)

        # create punctuated prediction
        punct_text = self.punctuate_texts(combined_preds)
        return punct_text

    def predict(self, input_slice):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        predictions, raw_outputs = self.model.predict([input_slice])
        return predictions, raw_outputs

    @staticmethod
    def split_on_toks(text, max_length=512, overlap=30):
        tokenizer = AutoTokenizer.from_pretrained("felflare/bert-restore-punctuation")
        
        # Tokenize the text with offsets
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding.tokens()
        offsets = encoding.offset_mapping
        
        total_tokens = len(tokens)
        resp = []
        start_token_idx = 0

        while start_token_idx < total_tokens:
            end_token_idx = min(start_token_idx + max_length, total_tokens)

            # Extract tokens and offsets for the chunk
            chunk_tokens = tokens[start_token_idx:end_token_idx]
            chunk_offsets = offsets[start_token_idx:end_token_idx]

            # Handle overlap
            if end_token_idx < total_tokens:
                overlap_end_idx = min(end_token_idx + overlap, total_tokens)
            else:
                overlap_end_idx = end_token_idx

            chunk_tokens_with_overlap = tokens[start_token_idx:overlap_end_idx]
            chunk_offsets_with_overlap = offsets[start_token_idx:overlap_end_idx]

            # Reconstruct chunk text
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens_with_overlap)

            # Get character indices
            chunk_start_char = chunk_offsets[0][0]
            chunk_end_char = chunk_offsets[end_token_idx - start_token_idx - 1][1]

            resp_obj = {
                "text": chunk_text,
                "start_idx": chunk_start_char,
                "end_idx": chunk_end_char
            }
            resp.append(resp_obj)

            start_token_idx = end_token_idx  # Move to next chunk

        logging.info(f"Sliced transcript into {len(resp)} slices.")
        return resp

    @staticmethod
    def combine_results(full_text: str, text_slices, tokenizer):
        """
        Given a full text and predictions of each slice combines predictions into a single text again.
        Performs validataion wether text was combined correctly
        """
        # Use the tokenizer to split full_text
        split_full_text = tokenizer.tokenize(full_text)
        split_full_text_len = len(split_full_text)
        output_text = []
        index = 0

        if len(text_slices[-1]) <= 3 and len(text_slices) > 1:
            text_slices = text_slices[:-1]

        for _slice in text_slices:
            slice_wrds = len(_slice)
            for ix, wrd in enumerate(_slice):
                if index == split_full_text_len:
                    break

                token = list(wrd.keys())[0]

                if split_full_text[index] == token and \
                        ix <= slice_wrds - 3 and text_slices[-1] != _slice:
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
                elif split_full_text[index] == token and text_slices[-1] == _slice:
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)

        # Debug statements to compare tokens
        # print("Tokens from output_text:", [i[0] for i in output_text])
        # print("Tokens from split_full_text:", split_full_text)

        assert [i[0] for i in output_text] == split_full_text
        return output_text


    @staticmethod
    def punctuate_texts(full_pred: list):
        """
        Given a list of Predictions from the model, applies the predictions to text,
        thus punctuating it.
        """
        punct_resp = ""
        for i in full_pred:
            word, label = i
            if label[-1] == "U":
                punct_wrd = word.capitalize()
            else:
                punct_wrd = word

            if label[0] != "O":
                punct_wrd += label[0]

            punct_resp += punct_wrd + " "
        punct_resp = punct_resp.strip()
        # Append trailing period if doesnt exist.
        if punct_resp[-1].isalnum():
            punct_resp += "."
        return punct_resp


if __name__ == "__main__":
    punct_model = RestorePuncts()
    # read test file
    with open('../tests/sample_text.txt', 'r') as fp:
        test_sample = fp.read()
    # predict text and print
    punctuated = punct_model.punctuate(test_sample)
    print(punctuated)
