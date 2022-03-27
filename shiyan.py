import pickle
from dpr.models import init_reader_components

def read_serialized_data_from_files(paths):
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            data = pickle.load(reader)
            results.extend(data)
    return results

if __name__ == '__main__':
    tensorizer, reader, optimizer = init_reader_components(cfg.encoder.encoder_model_type, cfg)
    passage_text = "row 1 is : Year is 1958 ; Title is Paul Anka ; Label is ABC Paramount ; Format is LP ; US is - ; Certifications is - . row 2 is : Year is 1959 ; Title is My Heart Sings ; Label is ABC Paramount ; Format is CD , LP ; US is - ; Certifications is - . row 3 is : Year is 1960 ; Title is Swings For Young Lovers ; Label is ABC Paramount ; Format is CD , LP ; US is - ; Certifications is - . row 4 is : Year is 1961 ; Title is It 's Christmas Everywhere ; Label is ABC Paramount ; Format is LP ; US is - ; Certifications is - . row 5 is : Year is 1962 ; Title is Young , Alive and In Love ! ; Label is RCA Victor ; Format is LP ; US is 61 ; Certifications is - . row 6 is : Year is 1962 ; Title is Let 's Sit This One Out ; Label is RCA Victor ; Format is LP ; US is 137 ; Certifications is - . row 7 is : Year is 1963 ; Title is 3 Great Guys ( Paul Anka , Sam Cooke and Neil Sedaka ) ; Label is RCA Victor ; Format is LP ; US is - ; Certifications is - . row 8 is : Year is 1963 ; Title is Our Man Around The World ; Label is RCA Victor ; Format is LP ; US is - ; Certifications is - . row 9 is : Year is 1963 ; Title is Italiano ; Label is RCA Victor ; Format is LP ; US is - ; Certifications is - . row 10 is : Year is 1968 ; Title is Goodnight My Love ; Label is RCA Victor ; Format is LP ; US is 101 ; Certifications is - . row 11 is : Year is 1969 ; Title is Life Goes On ; Label is RCA Victor ; Format is LP ; US is 194 ; Certifications is - . row 12 is : Year is 1972 ; Title is Paul Anka ; Label is Buddah ; Format is CD , LP ; US is 188 ; Certifications is - . row 13 is : Year is 1972 ; Title is Jubilation ; Label is Buddah ; Format is CD , LP ; US is 192 ; Certifications is - . row 14 is : Year is 1974 ; Title is Anka ; Label is United Artists ; Format is CD , LP ; US is 9 ; Certifications is Gold . row 15 is : Year is 1975 ; Title is Feelings ; Label is United Artists ; Format is CD , LP ; US is 36 ; Certifications is - . row 16 is : Year is 1975 ; Title is Times of Your Life ( 9 of 10 cuts from previous 2 albums ) ; Label is United Artists ; Format is LP ; US is 22 ; Certifications is Gold . row 17 is : Year is 1976 ; Title is The Painter ; Label is United Artists ; Format is CD , LP ; US is 85 ; Certifications is - . row 18 is : Year is 1977 ; Title is The Music Man ; Label is United Artists ; Format is LP ; US is 195 ; Certifications is - . row 19 is : Year is 1978 ; Title is Listen to Your Heart ; Label is RCA Victor ; Format is CD , LP ; US is 179 ; Certifications is - . row 20 is : Year is 1979 ; Title is Headlines ; Label is RCA Victor ; Format is CD , LP ; US is - ; Certifications is - ."
    passage_token_ids = tensorizer.text_to_tensor(passage_text, add_special_tokens=False)
    answer_spans = [
        _find_answer_positions(passage_token_ids, answers_token_ids[i]) for i in range(len(answers))
    
    # flatten spans list
    answer_spans = [item for sublist in answer_spans for item in sublist]
    answers_spans = list(filter(None, answer_spans))
    ctx.answers_spans = answers_spans